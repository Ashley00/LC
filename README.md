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

#### Day02

#### 209 Minimum Size Subarray
1.滑动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。从而将O(n^2)暴力解法降为O(n).
2.每个元素在滑动窗后进来操作一次，出去操作一次，每个元素都是被操作两次，所以时间复杂度是 2 × n 也就是O(n)
3.暴力解法两个for循环
```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
```
```
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```
```
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, r = 0, 0 # two pointer
        result = 0
        summ = 0
        currLength = 0
        while r < len(nums):
            if summ < target:
                summ += nums[r]
                r += 1
                currLength += 1
            
            while summ >= target:
                summ -= nums[l]
                l += 1
                
                if result == 0: # no subarray yet
                    result = currLength
                else:
                    result = min(result, currLength)
                currLength -= 1
        return result
```
TC:O(n)  SC:O(1)

#### 59. Spiral Matrix II
1.循环不变量：左闭右开
```
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
```
```
def generateMatrix(self, n: int) -> List[List[int]]:
        result = [[0 for _ in range(n)] for _ in range(n)]
        round = n // 2 # number of loop we need
        i = 0
        num = 1

        for i in range(round):
            for j in range(i, n-i-1): # top line
                result[i][j] = num
                num += 1
            for j in range(i, n-i-1): # right line
                result[j][n-i-1] = num
                num += 1
            for j in range(n-i-1, i, -1): # bottom line
                result[n-i-1][j] = num
                num += 1
            for j in range(n-i-1, i, -1): # left line
                result[j][i] = num
                num += 1
        
        if n % 2 == 1: # odd n need to fill the middle
            result[round][round] = num
        return result
```
TC:O(n^2)  SC:O(1)

#### Interval Sum
1. 暴力解法：Worst Case: O(n*m) 查询m次，每次查询的范围都是从0 到 n - 1
2. 前缀和(Prefix Sum): 涉及计算区间和的问题时非常有用, 特别注意求解区间
```
def intervalSum(self, nums: List[int], intervals: List[List[int]]) -> List[int]:
        result = []
        prefix = []
        currSum = 0
        for n in nums:
            currSum += n
            prefix.append(currSum)
        
        for s, e in intervals:
            if s == 0:
                result.append(prefix[e])
            else:
                result.append(prefix[e] - prefix[s-1])

        return result
```
TC:O(n)  SC:O(n)

#### Purchase Land
1. Prefix Sum: column prefix and row prefix
```
def purchaseLand(self, vec: List[List[int]]) -> int:
        result = inf("float")
        n = len(vec)
        m = len(vec[0])
        horizontal_prefix = [0] * n
        vertical_prefix = [0] * m

        # calculate prefix for horizontal
        horizontal_sum = 0
        for i in range(n):
            horizontal_sum += sum(vec[i])
            horizontal_prefix[i] = horizontal_sum

        # calculate prefix for vertical
        vertical_sum = 0
        for j in range(m):
            summ = 0
            for i in range(n):
                summ += vec[i][j]
            vertical_sum += summ
            vertical_prefix = vertical_sum
        
        # calculate min diff
        total_sum = horizontal_prefix[n-1]
        result = inf("float")
        for i in range(n):
            result = min(result, abs(total_sum - horizontal_prefix[i] * 2))
        for j in range(m):
            result = min(result, abs(total_sum - vertical_prefix[j] * 2))
        return result
```
TC:O(n^2)  SC:O(n)
![image](https://github.com/user-attachments/assets/3d10324b-caeb-4f3e-9574-6d7d348cd660)
