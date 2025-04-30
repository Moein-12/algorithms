"""This module implements various merge sorting algorithms and utilities for merging sequences.
It provides both in-place and iterator-based merge operations with support for custom
comparison keys and k-way merging.

Key Features:
- In-place merge sort implementation
- K-way merge sort with iterator support
- Stable sorting (preserves order of equal elements)
- Custom key functions for comparisons
- Memory efficient merge operations

The module includes the following main functions:
- merge_in_place: Merges two sorted portions of a sequence in-place
- standard_merge_sort: Classic recursive merge sort implementation
- merge: Merges multiple sorted iterators into a single sorted generator
- merge_sorted: Performs k-way merge sort on a sequence
"""

from collections import abc
from itertools import pairwise
import typing as tp

from ._utilities import Key, UNDEFINED, identity


T = tp.TypeVar('T') 

# standard definition
def merge_in_place(seq: abc.MutableSequence[T], /, start: int, middle: int, end: int, *, key: Key[T]):
    """Merge two sorted portions of a sequence in-place.

    This function merges two adjacent sorted portions of a sequence into a single sorted portion.
    The merge is performed in-place, modifying the original sequence.

    Args:
        seq (abc.MutableSequence[T]): The sequence containing the portions to merge.
        start (int): The starting index of the first portion (inclusive).
        middle (int): The ending index of the first portion and starting index of the second portion.
        end (int): The ending index of the second portion (exclusive).
        key (Key[T]): A function that extracts a comparison key from an element.

    Notes:
        - The portions [start:middle] and [middle:end] must be sorted according to the key function
          for the merge to produce correct results.
        - The merge operation is stable - equal elements maintain their relative order.
        - Uses O(n) extra space for temporary arrays.
        - Time complexity is O(n) where n is the total length of the portions being merged.
        - The original sequence is modified in-place.

    Example:
        >>> seq = [1, 3, 5, 2, 4, 6]  # Two sorted portions: [1,3,5] and [2,4,6]
        >>> merge_in_place(seq, 0, 3, 6)
        >>> seq
        [1, 2, 3, 4, 5, 6]
    """
    # adapted from https://www.geeksforgeeks.org/in-place-merge-sort/

    n1 = middle - start
    n2 = end - middle

    # Create temp arrays
    L = [UNDEFINED] * n1
    R = [UNDEFINED] * n2

    # Copy data to temp arrays L[] and R[]
    for i in range(n1):
        L[i] = seq[start + i]
    for j in range(n2):
        R[j] = seq[middle + j]

    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = start  # Initial index of merged subarray

    # Merge the temp arrays back
    # into arr[start..right]
    while i < n1 and j < n2:
        if key(R[j]) < key(L[i]):
            seq[k] = L[i]
            i += 1
        else:
            seq[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[],
    # if there are any
    while i < n1:
        seq[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], 
    # if there are any
    while j < n2:
        seq[k] = R[j]
        j += 1
        k += 1

def standard_merge_sort(
    seq: abc.MutableSequence[T], /, start: int, end: int, *, key: Key[T] = identity
):
    """Sort a mutable sequence in-place using the standard merge sort algorithm.

    This function implements the classic merge sort algorithm, recursively dividing
    the sequence into halves and merging them back in sorted order.

    Args:
        seq: The mutable sequence to sort.
        start: The starting index of the range to sort (inclusive).
        end: The ending index of the range to sort (exclusive).
        key: A function that extracts a comparison key from an element.
            Defaults to the identity function.

    Raises:
        IndexError: If the range [start, end) is invalid (i.e., if start >= end).

    Notes:
        - The sort is stable - equal elements maintain their relative order
        - Uses O(n) extra space for merging
        - Time complexity is O(n log n)
        - The sequence is modified in-place
    """
    if start + 1 < end:
        middle = (start + end) >> 1  # equivalent to (start + end) // 2

        standard_merge_sort(seq, start, middle, key=key)
        standard_merge_sort(seq, middle, end, key=key)
        merge_in_place(seq, start, middle, end, key=key)

    # elif start + 1 == end:
    #     pass  # as it's already sorted
    
    else:
        raise IndexError(f'Invalid range: [{start}, {end})')

def _merged(
    gens: abc.MutableSequence[abc.Iterator[T]],
    key: Key[T] = None,
) -> abc.Generator[T]:
    """Merge multiple sorted iterators into a single sorted generator.
    
    Args:
        gens: A sequence of sorted iterators to merge. Each iterator should yield
            elements in sorted order according to the key function.
        key: Optional function that returns the value to sort by for each element.
            If None, elements are compared directly using their natural ordering.

    Returns:
        A generator yielding elements from all input iterators in sorted order.
        The elements are yielded one at a time in ascending order based on the
        key function.

    Note:
        The input iterators must already be sorted according to the key function
        for the merge to produce correct results. The merge operation is stable,
        meaning that the relative order of equal elements is preserved.
        
    Implementation Details:
        - Uses a buffer (items) to store the current element from each iterator
        - Maintains an active count of non-exhausted iterators
        - For each iteration, finds the minimum element among active iterators
        - When an iterator is exhausted (returns UNDEFINED), it is moved to the end
            and excluded from future comparisons
        - The process continues until all iterators are exhausted
        
    Time Complexity:
        O(N * log(k)) where N is total number of elements and k is number of iterators
        
    Space Complexity:
        O(k) where k is the number of input iterators
    """
    
    items = [next(gen, UNDEFINED) for gen in gens]
    active_count = len(items)
    get_key = items.__getitem__ if key is None else lambda i: key(items[i])  # type: tp.Any
    
    for i in range(active_count - 1, -1, -1):
        if items[i] is UNDEFINED:
            items[i], items[active_count - 1] = items[active_count - 1], items[i]
            gens[i], gens[active_count - 1] = gens[active_count - 1], gens[i]
            active_count -= 1
    
    while active_count > 0:
        min_index = min(range(active_count), key=get_key)
        yield items[min_index]

        next_value = next(gens[min_index], UNDEFINED)

        if next_value is UNDEFINED:
            active_count -= 1
            items[min_index], items[active_count] = items[active_count], items[min_index]
            gens[min_index], gens[active_count] = gens[active_count], gens[min_index]
        else:
            items[min_index] = next_value
def _merge_sorted(
    seq: abc.Sequence[T],
    /,
    start: int,
    end: int,
    *,
    k: int = 2,
    key: Key[T] = identity,
) -> abc.Iterator[T]:
    """
    Recursively merges a slice of a sequence using a k-way merge strategy.

    Args:
        seq (abc.Sequence[T]): The input sequence to merge.
        start (int): The starting index of the slice to merge (inclusive).
        end (int): The ending index of the slice to merge (exclusive).
        k (int, optional): The number of partitions to split the slice into
          for merging. Defaults to 2.
        key (Callable[[T], Any], optional): A function to extract a comparison
          key from each element. Defaults to the identity function.

    Returns:
        abc.Iterator[T]: An iterator over the merged, sorted elements in the
          specified slice.

    Notes:
        - If the slice length is less than 2, returns an iterator over the
          slice as is.
        - Uses a divide-and-conquer approach, recursively splitting the
          slice into k parts and merging them.
    """
    if end - start < 2:
        return iter(seq[start:end])

    else:
        length = end - start
        return _merged(
            [_merge_sorted(seq, low, high, k=k, key=key) for low, high in pairwise(
                start + (length * ind // k) for ind in range(k + 1)
            )], key=key,
        )

def merge(
    gens: abc.MutableSequence[abc.Iterator[T]],
    key: Key[T] = None,
) -> abc.Generator[T]:
    """Merge multiple sorted iterators into a single sorted generator.

    Args:
        gens (abc.MutableSequence[abc.Iterator[T]]): A sequence of sorted iterators to merge.
        key (Key[T], optional): A function to extract a comparison key from each element.
            If None, elements are compared directly. Defaults to None.

    Returns:
        abc.Generator[T]: A generator yielding elements from all input iterators in sorted order.

    Raises:
        TypeError: If gens is not a sequence or if key is not callable.

    Notes:
        - The input iterators must already be sorted according to the key function
          for the merge to produce correct results.
        - This is a wrapper around the internal _merged function that adds input validation.
    """
    for check, typ, name in (
        (isinstance(gens, abc.Sequence), TypeError, 'gens must be a sequence'),
        (callable(key), TypeError, 'key must be callable'),
    ):
        if not check:
            raise typ(f'{name}, got {locals()[name.split()[0]]} instead')

    return _merged(gens, key=key)

def merge_sorted(
    seq: abc.Sequence[T],
    /,
    start: int = 0,
    end: int = None,
    *,
    k: int = 2,
    key: Key[T] = identity,
) -> abc.Iterator[T]:
    """Merge sort a sequence using k-way merging.

    This function performs a k-way merge sort on a sequence, allowing for efficient sorting
    of large sequences by dividing them into k partitions.

    Args:
        seq (abc.Sequence[T]): The sequence to sort.
        start (int, optional): The starting index of the range to sort. Defaults to 0.
        end (int, optional): The ending index (exclusive) of the range to sort.
            If None, uses the length of the sequence. Defaults to None.
        k (int, optional): Number of partitions to use in the k-way merge.
            Must be >= 2. Defaults to 2.
        key (Key[T], optional): Function to extract comparison key from elements.
            If None, compares elements directly. Defaults to identity function.

    Returns:
        abc.Iterator[T]: An iterator yielding the sorted elements.

    Raises:
        TypeError: If any of the following conditions are met:
            - seq is not a sequence
            - start is not an integer
            - end is not None or an integer
            - k is not an integer
            - key is not callable
        ValueError: If k is less than 2

    Examples:
        >>> list(merge_sorted([3, 1, 4, 1, 5, 9, 2, 6]))
        [1, 1, 2, 3, 4, 5, 6, 9]

        >>> list(merge_sorted([3, 1, 4, 1, 5], key=lambda x: -x))  # Sort in descending order
        [5, 4, 3, 1, 1]

        >>> list(merge_sorted([3, 1, 4, 1, 5], k=3))  # Use 3-way merging
        [1, 1, 3, 4, 5]
    """
    for check, typ, name in (
        (isinstance(seq, abc.Sequence), TypeError, 'seq must be a sequence'),
        (isinstance(start, int), TypeError, 'start must be an integer'),
        (end is None or isinstance(end, int), TypeError, 'end must be an integer or None'),
        (isinstance(k, int), TypeError, 'k must be an integer'),
        (callable(key), TypeError, 'key must be callable'),
        (k >= 2, ValueError, 'k must be at least 2'),
    ):
        if not check:
            raise typ(f'{name}, got {locals()[name.split()[0]]} instead')

    if end is None:
        end = len(seq)
    
    return _merge_sorted(seq, start, end, k=k, key=key)
