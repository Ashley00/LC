#### Day07
#### 454. 4Sum II
1. 分为ab, cd两组，跟two sum思路类似
```
def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        result = 0
        hashmap1 = {} # record sum from nums1 and nums2
        for n1 in nums1:
            for n2 in nums2:
                n = n1 + n2
                if n in hashmap1:
                    hashmap1[n] += 1
                else:
                    hashmap1[n] = 1
        # two sum logic
        for n3 in nums3:
            for n4 in nums4:
                diff = 0- (n3 + n4)
                if diff in hashmap1:
                    result += hashmap1[diff]
        return result
```
TC: O(n^2)  SC: O(n^2)

#### 383. Ransom Note
1. 杂志里面的字母不可重复使用。
```
def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        hashmap = {}
        for c in magazine:
            if c in hashmap:
                hashmap[c] += 1
            else:
                hashmap[c] = 1
            
        for c in ransomNote:
            if c in hashmap:
                if hashmap[c] == 0:
                    return False
                hashmap[c] -= 1
            else:
                return False
        return True
```
TC: O(n)  SC: O(1)

#### 15. 3Sum
1. 双指针法 要比哈希法高效一些,去重的操作中有很多细节需要注意
2. a,b,c (i,l,r)都要去重
```
def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)-2):
            # remove duplicates for i
            if i != 0 and nums[i] == nums[i-1]:
                continue
            l = i + 1
            r = len(nums) - 1
            while l < r:
                summ = nums[i] + nums[l] + nums[r]
                if summ == 0:
                    result.append([nums[i], nums[l], nums[r]])
                    r -= 1
                    l += 1
                    # remove duplicates for r
                    while l < r and nums[r] == nums[r+1]:
                        r -= 1
                    # remove duplicated for l
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                elif summ > 0:
                    r -= 1
                else:
                    l += 1
        return result
```
TC: O(n^2)  SC: O(1)

#### 18. 4Sum
```
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)-3):
            if i != 0 and nums[i] == nums[i-1]: # duplicates for i
                continue
            for j in range(i+1, len(nums)-2):
                if j != i+1 and nums[j] == nums[j-1]: # duplicates for j
                    continue
                l = j + 1
                r = len(nums) - 1
                while l < r:
                    summ = nums[i] + nums[j] + nums[l] + nums[r]
                    if summ == target:
                        result.append([nums[i],nums[j],nums[l],nums[r]])
                        l += 1
                        r -= 1
                        while l < r and nums[r] == nums[r+1]: # duplicates for r
                            r -= 1
                        while l < r and nums[l] == nums[l-1]: # duplicates for l
                            l += 1
                    elif summ > target:
                        r -= 1
                    else:
                        l += 1
        return result
```
TC: O(n^3)  SC: O(1)
