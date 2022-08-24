//! This library provides the `Rose` type, which is a data structure that has stable pointers.
//! It also happens to concurrent, which was a secondary goal of this project, because you can't
//! have a safe API without Atomics.
//!
//! # Example
//! ```
//! use rose_bloom::Rose;
//!
//! let rose = Rose::new();
//! let out1 = rose.push(1);
//! rose.push(2);
//! rose.push(3);
//! println!("{out1}");
//! ```
//!
#![deny(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![no_std]
use core::{
    alloc::Layout,
    mem::{self, ManuallyDrop},
    ops::Index,
    ptr::{self, addr_of, addr_of_mut, NonNull},
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};

extern crate alloc;
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
/// A lock-free growing element size linked list with stable pointers.
/// ```
/// use rose_bloom::Rose;
/// let rose = Rose::new();
/// let v = rose.push(1);
/// let _ = rose.push(2);
/// let _ = rose.push(3);
/// assert_eq!(rose[0], 3);
/// assert_eq!(rose[1], 2);
/// println!("{}", v); // prints 1
/// assert_eq!(rose[2], 1);
/// ```
pub struct Rose<T> {
    last: AtomicPtr<RoseInner<T>>,
}

#[repr(C)]
struct RoseInner<T> {
    next: *mut RoseInner<T>,
    length: AtomicUsize,
    capacity: usize,
    writers: AtomicUsize,
    data: [T; 0],
}

const fn next_multiple_of(n: usize, m: usize) -> usize {
    match n % m {
        0 => n,
        r => n + (m - r),
    }
}

impl<T> Rose<T> {
    const ALIGNMEMT: usize = mem::align_of::<RoseInner<T>>();

    #[track_caller]
    const fn size_assert(cap: usize) {
        assert!(next_multiple_of(Self::size(cap), Self::ALIGNMEMT) <= isize::MAX as usize);
    }

    const fn size(cap: usize) -> usize {
        mem::size_of::<RoseInner<T>>().saturating_add(mem::size_of::<T>().saturating_mul(cap))
    }
    const fn align() -> usize {
        mem::align_of::<RoseInner<T>>()
    }
    const fn layout(cap: usize) -> Layout {
        let size = Self::size(cap);
        Self::size_assert(size);
        // SAFETY: align_of returns a power of 2, and the size has been checked to be less than
        // isize::MAX.
        unsafe { Layout::from_size_align_unchecked(size, Self::ALIGNMEMT) }
    }

    fn alloc(capacity: usize) -> NonNull<RoseInner<T>> {
        // These could only be the case if someone is trying to allocate a rose with a never type,
        // which is incredibly stupid. But welcome to checking all invariants.
        // I also am not convinced that it would actually happen with repr(C) anyways, but
        // better be safe than sorry.
        // These should both compile out if they're false because they're all constants.
        assert_ne!(Self::align(), 0);
        assert!(Self::size(1) != 0);
        let layout = Self::layout(capacity);
        // SAFETY: Size is not zero, as tested above, so we can alloc.
        // We check for null pointer and handle the alloc error, so it must be non null.
        unsafe {
            let memory = alloc(layout);
            if memory.is_null() {
                handle_alloc_error(layout)
            }
            NonNull::new_unchecked(memory.cast::<RoseInner<T>>())
        }
    }
    /// Creates a new [`Rose<T>`].
    /// ```
    /// # use rose_bloom::Rose;
    /// let rose: Rose<i32> = Rose::new();
    /// ```
    #[must_use]
    pub fn new() -> Rose<T> {
        // Allocation code
        let memory = Self::alloc(1).as_ptr();
        // Initialization code
        // SAFETY: We are the only ones who can access this memory, so we can initialize it.
        unsafe {
            addr_of_mut!((*memory).next).write(ptr::null_mut());
            addr_of_mut!((*memory).length).write(AtomicUsize::new(0));
            addr_of_mut!((*memory).capacity).write(1);
            addr_of_mut!((*memory).writers).write(AtomicUsize::new(0));
        };
        Rose {
            last: AtomicPtr::new(memory),
        }
    }

    /// Push a `value` into the [`Rose`].
    /// ```
    /// # use rose_bloom::Rose;
    /// let rose = Rose::new();
    /// rose.push(1);
    /// ```
    /// 
    /// This is a O(1) operation.
    pub fn push(&self, value: T) -> &T {
        let item = ManuallyDrop::new(value);
        loop {
            let last = self.last.load(Ordering::Acquire);

            // Both of these are from the same `RoseInner`
            // SAFETY: Last is always a valid pointer.
            let length = unsafe { &(*last).length };
            let capacity = unsafe { (*last).capacity };
            // We only need access to this one piece of memory, so we use Relaxed.
            let index = length.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                if x == capacity {
                    // We need to allocate more.
                    None
                } else {
                    Some(x + 1)
                }
            });

            if let Ok(len) = index {
                // SAFETY: last is a valid pointer.
                // We use AcqRel to ensure that other threads are aware we're writing.
                let writer_count = unsafe { (*last).writers.fetch_add(1, Ordering::AcqRel) };
                // make sure `writers` isn't about to overflow
                assert!(
                    writer_count != usize::MAX,
                    "writing counter overflow, somehow we have usize::MAX writers"
                );
                // SAFETY: We now "own" the index, as the index was increased, so we can write to it.
                unsafe {
                    let out = ptr::addr_of_mut!((*last).data).cast::<T>().add(len);
                    out.write(addr_of!(item).cast::<T>().read());
                    // SAFETY: last is a valid pointer.
                    (*last).writers.fetch_sub(1, Ordering::AcqRel);
                    return &*out;
                }
            }

            let new = Self::alloc(capacity * 2);
            // SAFETY: We just created the pointer, so we can do whatever we want with it.
            let refer = unsafe {
                let new = new.as_ptr();
                addr_of_mut!((*new).next).write(last);
                addr_of_mut!((*new).length).write(AtomicUsize::new(1));
                addr_of_mut!((*new).capacity).write(capacity * 2);
                addr_of_mut!((*new).writers).write(AtomicUsize::new(0));
                let out = ptr::addr_of_mut!((*new).data).cast::<T>();
                out.write(addr_of!(item).cast::<T>().read());
                &*out
            };
            let success = self
                .last
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |x| {
                    if x == last {
                        Some(new.as_ptr())
                    } else {
                        // Somebody changed our pointer.
                        None
                    }
                });
            if success.is_ok() {
                return refer;
            }
            // Another thread updated the pointer before us.
            // SAFETY: We just created the pointer via alloc, so we can deallocate it.
            unsafe {
                dealloc(new.as_ptr().cast::<u8>(), Self::layout(capacity * 2));
            };
        }
    }

    /// Returns the length of this [`Rose<T>`].
    /// 
    /// This is an O(1) operation.
    pub fn len(&self) -> usize {
        let last = self.last.load(Ordering::Acquire);
        // SAFETY: last is a valid pointer.
        let length = unsafe { (*last).length.load(Ordering::Acquire) };
        let capacity = unsafe { (*last).capacity };
        let total_capacity = (capacity + 1).next_power_of_two() - 1;
        total_capacity - capacity + length
    }
    /// Are there elements in the [`Rose<T>`]?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    

}

/// This index is the number from the back.
impl<T> Index<usize> for Rose<T> {
    type Output = T;

    fn index(&self, mut index: usize) -> &Self::Output {
        let mut current = self.last.load(Ordering::Acquire);
        while !current.is_null() {
            // SAFETY: the pointer is always valid
            // length - writers is the number of valid writers.
            let length = unsafe {
                (*current).length.load(Ordering::Acquire)
                    - (*current).writers.load(Ordering::Acquire)
            };
            if length > index {
                // Great, we found it
                // SAFETY: add is in bounds because length > index
                return unsafe {
                    &*(addr_of!((*current).data)
                        .cast::<T>()
                        .add(length - index - 1))
                };
            }
            // If we didn't find it, go to the next list item.
            index -= length;
            current = unsafe { (*current).next };
        }
        panic!("Index {index} doesn't exist.")
    }
}
impl<T> Drop for Rose<T> {
    fn drop(&mut self) {
        let mut current = self.last.load(Ordering::Acquire);
        while !current.is_null() {
            // SAFETY: current is valid, data will have valid data for `length` items
            let length = unsafe { (*current).length.load(Ordering::Acquire) };
            let slice = unsafe {
                core::slice::from_raw_parts_mut(
                    ptr::addr_of_mut!((*current).data).cast::<T>(),
                    length,
                )
            };
            // Drop items
            // SAFETY: slice is a valid slice
            unsafe { ptr::drop_in_place(slice) };
            let next = unsafe { (*current).next };
            // Dealloc Rose
            // Current is a pointer allocated by the GlobalAlloc.
            unsafe { dealloc(current.cast::<u8>(), Self::layout((*current).capacity)) }

            current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        use alloc::boxed::Box;
        let rose = Rose::new();

        assert!(rose.is_empty());
        rose.push(Box::new(1));
        assert_eq!(rose.len(), 1);
        rose.push(Box::new(2));
        assert_eq!(rose.len(), 2);
        rose.push(Box::new(3));
        assert_eq!(*rose[0], 3);
        assert_eq!(*rose[1], 2);
        assert_eq!(*rose[2], 1);
        assert_eq!(rose.len(), 3);
    }

    #[test]
    fn dataraces() {
        extern crate std;
        use std::thread;
        let rose = Rose::new();
        rose.push(0);
        thread::scope(|s| {
            s.spawn(|| {
                let _ = rose[0];
                rose.push(1);
            });
            s.spawn(|| {
                let _ = rose[0];
                rose.push(2);
            });
        });
    }
}
