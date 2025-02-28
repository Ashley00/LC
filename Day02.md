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
