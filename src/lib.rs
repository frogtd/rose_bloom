#![forbid(unsafe_op_in_unsafe_fn)]
use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem,
    ptr::{self, addr_of_mut, NonNull},
};
pub struct Rose<T> {
    last: NonNull<RoseInner<T>>,
}

unsafe impl<T> Send for Rose<T> where T: Send {}
unsafe impl<T> Sync for Rose<T> where T: Sync {}

#[repr(C)]
struct RoseInner<T> {
    capacity: usize,
    length: usize,
    next: *mut RoseInner<T>,
    data: [T; 0],
}
impl<T> Clone for RoseInner<T> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            length: self.length,
            next: self.next,
            data: [],
        }
    }
}
impl<T> std::fmt::Debug for RoseInner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoseInner")
            .field("capacity", &self.capacity)
            .field("length", &self.length)
            .field("next", &self.next)
            .finish()
    }
}
#[derive(Debug)]
pub struct Petal<'a, T>(*const T, PhantomData<&'a T>);

// Note: do these require a T: Send/Sync bound?
unsafe impl<'a, T> Send for Petal<'a, T> {}
unsafe impl<'a, T> Sync for Petal<'a, T> {}

const fn next_multiple_of(n: usize, m: usize) -> usize {
    match n % m {
        0 => n,
        r => n + (m - r),
    }
}

impl<T> Rose<T> {
    const ALIGNMEMT: usize = mem::align_of::<RoseInner<T>>();

    #[track_caller]
    const fn size_assert(n: usize) {
        assert!(next_multiple_of(Self::size(n), Self::ALIGNMEMT) <= isize::MAX as usize);
    }

    const fn size(n: usize) -> usize {
        mem::size_of::<RoseInner<T>>().saturating_add(mem::size_of::<T>().saturating_mul(n))
    }
    const fn align() -> usize {
        mem::align_of::<RoseInner<T>>()
    }
    const fn layout(n: usize) -> Layout {
        let size = Self::size(n);
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
            let memory = alloc::alloc(layout);
            if memory.is_null() {
                alloc::handle_alloc_error(layout)
            }
            NonNull::new_unchecked(memory as *mut RoseInner<T>)
        }
    }
    pub fn new() -> Rose<T> {
        // Allocation code
        let memory = Self::alloc(1);
        // Initialization code
        unsafe {
            let memory = memory.as_ptr();
            addr_of_mut!((*memory).capacity).write(1);
            addr_of_mut!((*memory).length).write(0);
            addr_of_mut!((*memory).next).write(ptr::null_mut());
        };
        Rose { last: memory }
    }

    pub fn push(&mut self, value: T) -> &T {
        // SAFETY: Last is always a valid pointer.
        let capacity = unsafe { (*self.last.as_ptr()).capacity };
        let length = unsafe { (*self.last.as_ptr()).length };
        if capacity != length {
            // We have space.
            unsafe {
                let out = ptr::addr_of_mut!((*self.last.as_ptr()).data)
                    .cast::<T>()
                    .add((*self.last.as_ptr()).length);
                out.write(value);
                (*self.last.as_ptr()).length += 1;
                &*out
            }
        } else {
            // We need to alloc more.
            let new = Self::alloc(capacity * 2);
            let last = self.last;
            unsafe {
                let new = new.as_ptr();
                ptr::addr_of_mut!((*new).capacity).write(capacity * 2);
                ptr::addr_of_mut!((*new).length).write(0);
                ptr::addr_of_mut!((*new).next).write(last.as_ptr());
            }
            // We have setup the new pointer.
            self.last = new;
            // Now we have space.
            unsafe {
                let out = ptr::addr_of_mut!((*self.last.as_ptr()).data)
                    .cast::<T>()
                    .add((*self.last.as_ptr()).length);
                out.write(value);
                (*self.last.as_ptr()).length += 1;
                &*out
            }
        }
    }
}

impl<T> Drop for Rose<T> {
    fn drop(&mut self) {
        let mut current = self.last.as_ptr();
        while !current.is_null() {
            println!("DATA: \"{:?}\"", unsafe { (*current).clone() });

            // SAFETY: current is valid, data will have valid data for `length` items
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    ptr::addr_of_mut!((*current).data).cast::<T>(),
                    (*current).length,
                )
            };
            // Drop items
            // SAFETY: slice is a valid slice
            unsafe { ptr::drop_in_place(slice) };
            let next = unsafe { (*current).next };
            // Dealloc Rose
            // Current is a pointer allocated by the GlobalAlloc.
            unsafe { alloc::dealloc(current.cast::<u8>(), Self::layout((*current).capacity)) }

            current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut rose: Rose<Box<i32>> = Rose::new();
        rose.push(Box::new(1));
        rose.push(Box::new(5));
    }
}
