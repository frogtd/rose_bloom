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
    cmp,
    fmt::{self, Debug, Formatter},
    hash::{Hash, Hasher},
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
    ptr::{self, addr_of, addr_of_mut, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};
/// Iterator types.
pub mod iter;

extern crate alloc;
use alloc::{
    alloc::{alloc, dealloc, handle_alloc_error},
    vec::Vec,
};
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
    /// The next node in the list.
    next: *mut RoseInner<T>,
    // The number of valid elements in the array. It is less than or equal to the capacity and
    //  is a power of two.
    length: AtomicUsize,
    /// The capacity is the number of elements that can be stored in the array. It is always
    /// greater than or equal to one and is a power of two.
    capacity: usize,
    /// The number of writers currently writing to the array. It is always less than or equal
    /// to length.
    writers: AtomicUsize,
    /// The data. This is a ZST but you can just peer off the end of the struct into the elements.
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
        // SAFETY: Size is not zero, because RoseInner<T> is at least size_of::<usize>() * 4, so
        // we can alloc.
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
    /// # Panics
    /// Panics if there are `usize::MAX` concurrent writes happening at once.
    pub fn push(&self, value: T) -> &T {
        let mut item = ManuallyDrop::new(value);
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
                    let out = addr_of_mut!((*last).data).cast::<T>().add(len);
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
                let out = addr_of_mut!((*new).data).cast::<T>();
                out.write(addr_of_mut!(item).cast::<T>().read());
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
    /// ```
    /// # use rose_bloom::Rose;
    /// let rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        let last = self.last.load(Ordering::Acquire);
        // SAFETY: last is a valid pointer.
        let length = unsafe { (*last).length.load(Ordering::Acquire) };
        let capacity = unsafe { (*last).capacity };
        let total_capacity = (capacity + 1).next_power_of_two() - 1;
        total_capacity - capacity + length
    }
    /// Returns the capacity of this [`Rose<T>`].
    /// ```
    /// # use rose_bloom::Rose;
    /// let rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert!(rose.capacity() >= 3);
    /// ```
    pub fn capacity(&self) -> usize {
        let last = self.last.load(Ordering::Acquire);
        // SAFETY: last is a valid pointer.
        let capacity = unsafe { (*last).capacity };
        (capacity + 1).next_power_of_two() - 1
    }
    /// Are there elements in the [`Rose<T>`]?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Iterate over the elements of the [`Rose<T>`].
    /// ```
    /// # use rose_bloom::Rose;
    /// let rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// for i in rose.iter() {
    ///    println!("{}", i);
    /// }
    /// ```
    pub fn iter(&self) -> iter::Iter<'_, T> {
        self.into_iter()
    }
    /// Iterate over the elements of the [`Rose<T>`], allowing modification.
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// for i in rose.iter_mut() {
    ///    *i += 1;
    /// }
    /// ```
    /// This is an O(n) operation.
    pub fn iter_mut(&mut self) -> iter::IterMut<'_, T> {
        self.into_iter()
    }
    /// Removes the last element from the [`Rose<T>`] and returns it, or `None` if it is empty.
    ///
    /// This is an O(1) operation.
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.pop(), Some(3));
    /// assert_eq!(rose.pop(), Some(2));
    /// assert_eq!(rose.pop(), Some(1));
    /// assert_eq!(rose.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        let mut current = NonNull::new(*self.last.get_mut())?;
        let length = unsafe { *(*current.as_ptr()).length.get_mut() };
        if length == 0 {
            // Time to deallocate the current node and move on to the next one.
            let next = unsafe { (*current.as_ptr()).next };
            let capacity = unsafe { (*current.as_ptr()).capacity };
            unsafe {
                dealloc(
                    self.last.get_mut().cast::<u8>(),
                    Rose::<T>::layout(capacity),
                );
            }
            self.last = AtomicPtr::new(next);
            current = NonNull::new(*self.last.get_mut())?;
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

    /// Returns a reference to an element, or `None` if the index is out of bounds.
    /// This indexes from the back of the [`Rose<T>`] to the front.
    /// This is an O(log n) operation.
    /// # Examples
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.get(0), Some(&3));
    /// assert_eq!(rose.get(1), Some(&2));
    /// assert_eq!(rose.get(2), Some(&1));
    /// assert_eq!(rose.get(3), None);
    /// ```
    pub fn get(&self, mut index: usize) -> Option<&T> {
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
                return Some(unsafe {
                    &*(addr_of!((*current).data)
                        .cast::<T>()
                        .add(length - index - 1))
                });
            }
            // If we didn't find it, go to the next list item.
            index -= length;
            current = unsafe { (*current).next };
        }
        None
    }
    /// Returns a mutable reference to an element, or `None` if the index is out of bounds.
    /// This indexes from the back of the [`Rose<T>`] to the front.
    /// This is an O(log n) operation.
    /// # Examples
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.get_mut(0), Some(&mut 3));
    /// assert_eq!(rose.get_mut(1), Some(&mut 2));
    /// assert_eq!(rose.get_mut(2), Some(&mut 1));
    /// assert_eq!(rose.get_mut(3), None);
    /// ```
    pub fn get_mut(&mut self, mut index: usize) -> Option<&mut T> {
        let mut current = *self.last.get_mut();
        while !current.is_null() {
            // SAFETY: the pointer is always valid
            // length is the number of valid areas, and there are no active writers beacuse we take
            // &mut
            let length = unsafe { *(*current).length.get_mut() };
            if length > index {
                // Great, we found it
                // SAFETY: add is in bounds because length > index
                return Some(unsafe {
                    &mut *(addr_of_mut!((*current).data)
                        .cast::<T>()
                        .add(length - index - 1))
                });
            }
            // If we didn't find it, go to the next list item.
            index -= length;
            current = unsafe { (*current).next };
        }
        None
    }

    /// Removes an element from the [`Rose<T>`] and returns it, or `None` if it is empty.
    /// This indexes from the back of the [`Rose<T>`] to the front.
    ///
    /// This is an O(n) operation. Avoid using this if possible. It is better to use [`Self::pop`].
    /// # Examples
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(Box::new(1));
    /// rose.push(Box::new(2));
    /// rose.push(Box::new(3));
    /// assert_eq!(rose.remove(2), Some(Box::new(1)));
    /// ```
    pub fn remove(&mut self, mut index: usize) -> Option<T> {
        if self.len() <= index {
            // We can't remove anything if the index is out of bounds.
            return None;
        }
        let mut current = self.last.load(Ordering::Acquire);
        let mut prev: Option<T> = None;
        while !current.is_null() {
            // SAFETY: the pointer is always valid
            // length - writers is the number of valid places.
            let length = unsafe {
                (*current).length.load(Ordering::Acquire)
                    - (*current).writers.load(Ordering::Acquire)
            };

            if length > index {
                // Great, we found it
                // SAFETY: add is in bounds because length > index
                let elem_ptr = unsafe {
                    addr_of_mut!((*current).data)
                        .cast::<T>()
                        .add(length - index - 1)
                };
                let out = unsafe { elem_ptr.read() };
                unsafe { elem_ptr.add(1).copy_to(elem_ptr, length - index - 1) };
                unsafe {
                    *(*current).length.get_mut() -= 1;
                }

                if let Some(elem) = prev {
                    unsafe {
                        addr_of_mut!((*current).data)
                            .cast::<T>()
                            .add(*(*current).length.get_mut())
                            .write(elem);
                        *(*current).length.get_mut() += 1;
                    }
                }
                return Some(out);
            }
            // We need to move the list.
            let base_ptr = unsafe { addr_of_mut!((*current).data).cast::<T>() };
            let first = if length > 0 {
                let first = Some(unsafe { base_ptr.read() });
                // This doesn't overflow because length is always at least 1.
                let elements = length - 1;
                // We need to move elements forward 1.
                // SAFETY: add is in bounds because length > 0, and there are `elements` we need to
                // which was calculated via length.
                unsafe {
                    base_ptr.add(1).copy_to(base_ptr, elements);
                }
                // SAFETY: current is always valid and we have exclusive access to the list,
                // so we can use .get_mut().
                unsafe {
                    *(*current).length.get_mut() -= 1;
                };
                first
            } else {
                None
            };
            if let Some(elem) = prev {
                unsafe {
                    base_ptr.add(*(*current).length.get_mut()).write(elem);
                    *(*current).length.get_mut() += 1;
                }
            }
            prev = first;
            // If we didn't find it, go to the next list item.
            index -= length;
            current = unsafe { (*current).next };
        }
        unreachable!()
    }

    /// Push an element to the [`Rose<T>`].
    /// This takes `&mut self`, so can be marginally faster than [`Self::push`] and can return `&mut T`.
    ///
    /// This is an O(1) operation, amortized.
    /// # Examples
    /// ```
    /// # use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push_mut(1);
    /// rose.push_mut(2);
    /// rose.push_mut(3);
    /// assert_eq!(rose.len(), 3);
    /// ```
    pub fn push_mut(&mut self, value: T) -> &mut T {
        let current = *self.last.get_mut();
        let length = unsafe { *(*current).length.get_mut() };
        let capacity = unsafe { (*current).capacity };
        if length < capacity {
            // We have space.
            unsafe {
                let ptr = addr_of_mut!((*current).data).cast::<T>().add(length);
                ptr.write(value);
                *(*current).length.get_mut() += 1;
                &mut *ptr
            }
        } else {
            // We need to allocate
            let ptr = Self::alloc(capacity * 2);
            unsafe {
                let ptr = ptr.as_ptr();
                addr_of_mut!((*ptr).next).write(current);
                addr_of_mut!((*ptr).length).write(AtomicUsize::new(1));
                addr_of_mut!((*ptr).capacity).write(capacity * 2);
                addr_of_mut!((*ptr).writers).write(AtomicUsize::new(0));
                addr_of_mut!((*ptr).data).cast::<T>().write(value);
                *self.last.get_mut() = ptr;
                &mut *addr_of_mut!((*ptr).data).cast::<T>()
            }
        }
    }
    /// Returns a reference to the first element of the [`Rose<T>`].
    /// This is an O(log n) operation.
    /// # Examples
    /// ```
    /// use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.first(), Some(&1));
    /// ```
    pub fn first(&self) -> Option<&T> {
        self.get(self.len() - 1)
    }

    /// Returns a mutable reference to the first element of the [`Rose<T>`].
    /// This is an O(log n) operation.
    /// # Examples
    /// ```
    /// use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.first_mut(), Some(&mut 1));
    /// ```
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len() - 1)
    }

    /// Returns a reference to the last element of the [`Rose<T>`].
    /// This is an O(1) operation.
    /// # Examples
    /// ```
    /// use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.last(), Some(&3));
    /// ```
    pub fn last(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a mutable reference to the last element of the [`Rose<T>`].
    /// This is an O(1) operation.
    /// # Examples
    /// ```
    /// use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.last_mut(), Some(&mut 3));
    /// ```
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Clears the [`Rose<T>`], removing all values.
    /// This is an O(n) operation.
    /// # Examples
    /// ```
    /// use rose_bloom::Rose;
    /// let mut rose = Rose::new();
    /// rose.push(1);
    /// rose.push(2);
    /// rose.push(3);
    /// assert_eq!(rose.len(), 3);
    /// rose.clear();
    /// assert_eq!(rose.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        let mut current = *self.last.get_mut();
        let mut capacity = unsafe { (*current).capacity };
        while capacity != 1 {
            // SAFETY: current is valid, data will have valid data for `length` items
            let length = *unsafe { (*current).length.get_mut() };
            let slice = unsafe {
                core::slice::from_raw_parts_mut(addr_of_mut!((*current).data).cast::<T>(), length)
            };
            // Drop items
            // SAFETY: slice is a valid slice
            unsafe { ptr::drop_in_place(slice) };
            let next = unsafe { (*current).next };
            // Dealloc Rose
            // Current is a pointer allocated by the GlobalAlloc.
            unsafe { dealloc(current.cast::<u8>(), Self::layout((*current).capacity)) }
            current = next;
            capacity = unsafe { (*current).capacity };
        }
        // Capacity is 1, so we can just drop the last item, if it exists.
        let length = *unsafe { (*current).length.get_mut() };
        if length == 1 {
            unsafe {
                // Drop data.
                addr_of_mut!((*current).data).cast::<T>().read()
            };
            *unsafe { (*current).length.get_mut() } = 0;
        }

        self.last = AtomicPtr::new(current);
    }
}

/// This index is the number from the back.
/// This is a O(log n) operation.
impl<T> Index<usize> for Rose<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(x) => x,
            None => panic!("Index out of bounds, {index} is greater than {len}", len = self.len()),
        }
    }
}

/// This index is the number from the back.
/// This is a O(log n) operation.
impl<T> IndexMut<usize> for Rose<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if let Some(x) = self.get_mut(index) {
            // This transmute is purely to avoid a compiler error that will be fixed with 
            // polonius. I confirmed this code without the transmute works when using polonius.
            return unsafe { core::mem::transmute(x) };
        };
        panic!("Index out of bounds, {index} is greater than {len}", len = self.len())
    }
}
impl<T> Drop for Rose<T> {
    fn drop(&mut self) {
        let mut current = *self.last.get_mut();
        while !current.is_null() {
            // SAFETY: current is valid, data will have valid data for `length` items
            let length = *unsafe { (*current).length.get_mut() };
            let slice = unsafe {
                core::slice::from_raw_parts_mut(addr_of_mut!((*current).data).cast::<T>(), length)
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

impl<T> From<Vec<T>> for Rose<T> {
    fn from(mut vec: Vec<T>) -> Self {
        let rose = Self::new();
        for item in vec.drain(..) {
            rose.push(item);
        }
        rose
    }
}

impl<T> From<Rose<T>> for Vec<T> {
    fn from(rose: Rose<T>) -> Self {
        rose.into_iter().collect()
    }
}

impl<T> Clone for Rose<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let new = Self::new();
        for x in self {
            new.push(x.clone());
        }
        new
    }
}

impl<T> Debug for Rose<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Default for Rose<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Eq for Rose<T> where T: Eq {}
impl<T> PartialEq for Rose<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T> Hash for Rose<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut current = self.last.load(Ordering::Acquire);
        while !current.is_null() {
            let length = unsafe { (*current).length.load(Ordering::Acquire) };
            let slice = unsafe {
                core::slice::from_raw_parts(addr_of_mut!((*current).data).cast::<T>(), length)
            };
            T::hash_slice(slice, state);
            let next = unsafe { (*current).next };
            current = next;
        }
    }
}

impl<T> Extend<T> for Rose<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T> FromIterator<T> for Rose<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut rose = Self::new();
        rose.extend(iter);
        rose
    }
}
impl<T> Ord for Rose<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T> PartialOrd for Rose<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
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
        assert_eq!(rose.len(), 3);

        assert_eq!(*rose[0], 3);
        assert_eq!(*rose[1], 2);
        assert_eq!(*rose[2], 1);

        let mut iter = (&rose).into_iter();
        assert_eq!(iter.next(), Some(&Box::new(3)));
        assert_eq!(iter.next(), Some(&Box::new(2)));
        assert_eq!(iter.next(), Some(&Box::new(1)));
        assert_eq!(iter.next(), None);

        let mut iter = rose.into_iter();
        assert_eq!(iter.next(), Some(Box::new(3)));
        assert_eq!(iter.next(), Some(Box::new(2)));
        assert_eq!(iter.next(), Some(Box::new(1)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn mut_iter() {
        use alloc::boxed::Box;
        let mut rose = Rose::new();
        rose.push(Box::new(1));
        rose.push(Box::new(2));
        rose.push(Box::new(3));
        for x in rose.iter_mut() {
            **x += 1;
        }
        let mut iter = rose.iter_mut();
        assert_eq!(iter.next(), Some(&mut Box::new(4)));
        assert_eq!(iter.next(), Some(&mut Box::new(3)));
        assert_eq!(iter.next(), Some(&mut Box::new(2)));
        assert_eq!(iter.next(), None);
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
    #[test]
    #[should_panic]
    fn out_of_bounds() {
        let rose = Rose::new();
        rose.push(0);
        let _= &rose[1];
    }
    #[test]
    #[should_panic]
    fn out_of_bounds_mut() {
        let mut rose = Rose::new();
        rose.push(0);
        let _= &mut rose[1];
    }
}
