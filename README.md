#### Day01
#### 数组理论基础
1. 数组是存放在连续内存空间上的相同类型数据的集合。
2. 数组在内存空间的地址是连续的，所以删除或者增添元素的时候，就难免要移动其他元素的地址。数组的元素是不能删的，只能覆盖。
3. 在C++中二维数组在地址空间上是连续的。
4. Java是没有指针的，同时也不对程序员暴露其元素的地址，寻址操作完全交给虚拟机。
5. Python 没有build-in Array. List 可以当成Array用。

#### 704.Binary Search
1. 数组为有序数组，无重复元素。
2. 区间的定义就是不变量。

    左闭右闭：[left, right]，`while left <= right: left=mid+1, right=mid-1`
	
    左闭右开：[left, right)，`while left < right: left=mid+1, right=mid`
	
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
```
```
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
```
```python
def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        # target in [left, right]
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return -1
```
TC: O(logn)  SC: O(1)

#### 27. Remove element
1. 必须仅使用 O(1) 额外空间并原地修改输入数组.
2. Brute force: 两层for loop, outer loop 遍历数组，inner loop 更新数组。TC: O(n^2)  SC: O(1)
3. 双指针（快慢指针）：快指针-找不含目标的数组 慢指针-指向更新新数组的下标
```
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
```
```
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
```
```python
def removeElement(self, nums: List[int], val: int) -> int:
        fastIndex, slowIndex = 0, 0
        for fastIndex in range(len(nums)):
            if nums[fastIndex] != val:
                nums[slowIndex] = nums[fastIndex]
                slowIndex += 1
        return slowIndex
```
TC: O(n)  SC: O(1)

#### 977. Squares of a sorted array
1.数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。- 双指针
```
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
```
```python
def sortedSquares(self, nums: List[int]) -> List[int]:
        start, end = 0, len(nums)-1
        result = [0] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            s = nums[start] ** 2
            e = nums[end] ** 2
            if s >= e:
                result[i] = s
                start += 1
            else:
                result[i] = e
                end -= 1
        return result
```
TC: O(n)  SC: O(1)
