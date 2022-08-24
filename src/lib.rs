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
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};

extern crate alloc;
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
/// A growing element size linked list with stable pointers.
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
    capacity: usize,
    length: AtomicUsize,
    next: *mut RoseInner<T>,
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
        unsafe {
            let layout = Self::layout(capacity);
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
        unsafe {
            addr_of_mut!((*memory).capacity).write(1);
            addr_of_mut!((*memory).length).write(AtomicUsize::new(0));
            addr_of_mut!((*memory).next).write(ptr::null_mut());
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
    pub fn push(&self, value: T) -> &T {
        let item = ManuallyDrop::new(value);
        loop {
            // I just need to get access to some last element, it doesn't matter if it's out of date.
            // If it is, we will just try again.
            let last = self.last.load(Ordering::Relaxed);

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
                // We now "own" the index, as the index was increased.
                unsafe {
                    let out = ptr::addr_of_mut!((*last).data).cast::<T>().add(len);
                    out.write(addr_of!(item).cast::<T>().read());
                    return &*out;
                }
            }

            let new = Self::alloc(capacity * 2);

            let refer = unsafe {
                let new = new.as_ptr();
                ptr::addr_of_mut!((*new).capacity).write(capacity * 2);
                ptr::addr_of_mut!((*new).length).write(AtomicUsize::new(1));
                ptr::addr_of_mut!((*new).next).write(last);
                // We just created the pointer, so we can do whatever we want with it.
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
            // We failed to update the pointer, so we need to try again.
            unsafe {
                dealloc(new.as_ptr().cast::<u8>(), Self::layout(capacity * 2));
            };
        }
    }
}

/// This index is the number from the back.
impl<T> Index<usize> for Rose<T> {
    type Output = T;

    fn index(&self, mut index: usize) -> &Self::Output {
        let mut current = self.last.load(Ordering::Relaxed);
        while !current.is_null() {
            let length = unsafe { (*current).length.load(Ordering::Relaxed) };
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
        let mut current = self.last.load(Ordering::Relaxed);
        while !current.is_null() {
            // SAFETY: current is valid, data will have valid data for `length` items
            let length = unsafe { (*current).length.load(Ordering::Relaxed) };
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
        rose.push(Box::new(1));
        rose.push(Box::new(2));
        rose.push(Box::new(3));
        assert_eq!(*rose[0], 3);
        assert_eq!(*rose[1], 2);
        assert_eq!(*rose[2], 1);
    }
}
