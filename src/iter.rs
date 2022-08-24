use core::{
    marker::PhantomData,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::atomic::{AtomicPtr, Ordering},
};

use alloc::alloc::dealloc;

use crate::{Rose, RoseInner};
/// An borrowed iterator over the elements of a `Rose`.
pub struct Iter<'a, T: 'a> {
    rose: PhantomData<&'a Rose<T>>,
    pos: usize,
    ptr: *const RoseInner<T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let mut length = unsafe { (*self.ptr).length.load(Ordering::Acquire) };
        if length == self.pos {
            // Move on to the next node, we've run out here.
            self.ptr = unsafe { (*self.ptr).next };
            if self.ptr.is_null() {
                return None;
            }
            self.pos = 0;
            length = unsafe { (*self.ptr).length.load(Ordering::Acquire) };
        }

        let base_ptr = unsafe { addr_of!((*self.ptr).data).cast::<T>() };
        let index = length - self.pos - 1;
        self.pos += 1;
        Some(unsafe { &*base_ptr.add(index) })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.ptr.is_null() {
            return (0, Some(0));
        }
        let ptr = self.ptr;
        // SAFETY: ptr is a valid pointer if not null.
        let length = unsafe { (*ptr).length.load(Ordering::Acquire) };
        let capacity = unsafe { (*ptr).capacity };
        let total_capacity = (capacity + 1).next_power_of_two() - 1;
        let len = total_capacity - capacity + length;
        (len, Some(len))
    }
}

impl<'a, T> IntoIterator for &'a Rose<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            rose: PhantomData,
            pos: 0,
            ptr: self.last.load(Ordering::Acquire),
        }
    }
}
/// An iterator over the elements of a `Rose` that allows modification.
pub struct IterMut<'a, T: 'a> {
    rose: PhantomData<&'a mut Rose<T>>,
    pos: usize,
    ptr: *mut RoseInner<T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let mut length = unsafe { (*self.ptr).length.load(Ordering::Acquire) };
        if length == self.pos {
            // Move on to the next node, we've run out here.
            self.ptr = unsafe { (*self.ptr).next };
            if self.ptr.is_null() {
                return None;
            }
            self.pos = 0;
            length = unsafe { (*self.ptr).length.load(Ordering::Acquire) };
        }

        let base_ptr = unsafe { addr_of_mut!((*self.ptr).data).cast::<T>() };
        let index = length - self.pos - 1;
        self.pos += 1;
        Some(unsafe { &mut *base_ptr.add(index) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.ptr.is_null() {
            return (0, Some(0));
        }
        let ptr = self.ptr;
        // SAFETY: ptr is a valid pointer if not null.
        let length = unsafe { (*ptr).length.load(Ordering::Acquire) };
        let capacity = unsafe { (*ptr).capacity };
        let total_capacity = (capacity + 1).next_power_of_two() - 1;
        let len = total_capacity - capacity + length;
        (len, Some(len))
    }
}

impl<'a, T> IntoIterator for &'a mut Rose<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            rose: PhantomData,
            pos: 0,
            ptr: self.last.load(Ordering::Acquire),
        }
    }
}

/// An iterator over the elements of a `Rose`.
pub struct IntoIter<T> {
    rose: Rose<T>,
}

impl<T> IntoIterator for Rose<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { rose: self }
    }
}
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = NonNull::new(*self.rose.last.get_mut())?;
        let length = unsafe { *(*current.as_ptr()).length.get_mut() };
        if length == 0 {
            // Time to deallocate the current node and move on to the next one.
            let next = unsafe { (*current.as_ptr()).next };
            let capacity = unsafe { (*current.as_ptr()).capacity };
            unsafe {
                dealloc(
                    self.rose.last.get_mut().cast::<u8>(),
                    Rose::<T>::layout(capacity),
                );
            }
            self.rose.last = AtomicPtr::new(next);
            current = NonNull::new(*self.rose.last.get_mut())?;
        }
        // Current must have some length, because we either just checked that, or we went forward
        // to the next node, and the minimum capacity is 1, and because we just arrived at this
        // node, it must have at least one element because further back nodes are always
        //`capacity == length`.
        let index = unsafe {
            let index = *(*current.as_ptr()).length.get_mut() - 1;
            *(*current.as_ptr()).length.get_mut() -= 1;
            index
        };
        let base_ptr = unsafe { addr_of_mut!((*current.as_ptr()).data).cast::<T>() };

        Some(unsafe { base_ptr.add(index).read() })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.rose.len();
        (len, Some(len))
    }
}
