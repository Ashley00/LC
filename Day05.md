#### Day05
#### 哈希表理论基础
1. Hash Table: 哈希表是根据关键码的值而直接进行访问的数据结构.
2. 用来快速判断一个元素是否出现集合里.要枚举的话时间复杂度是O(n)，但如果使用哈希表的话， 只需要O(1)就可以做到。
3. 哈希函数，通过hashCode把名字转化为数值
4. 一般哈希碰撞有两种解决方法， 拉链法和线性探测法。
5. Separate chaining: Each hash table slot is associated with a linked list, where colliding elements are stored together in the list.
   拉链法就是要选择适当的哈希表的大小，这样既不会因为数组空值而浪费大量内存，也不会因为链表太长而在查找上浪费太多时间
6. Open addressing: When a collision happens, the hash table searches for the next available empty slot nearby to store the new data. 
   线性探测法，一定要保证tableSize大于dataSize。 我们需要依靠哈希表中的空位来解决碰撞问题
7. 3 types: array, set, map

#### 242. Valid Anagram
1. 定义一个数组叫做record用来记录字符串s里字符出现的次数.把字符映射到数组也就是哈希表的索引下标上
```
def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for c in s:
            ind = ord(c) - ord('a')
            record[ind] += 1
        for c in t:
            ind = ord(c) - ord('a')
            record[ind] -= 1
        for r in record:
            if r != 0:
                return False
        return True
```
Follow up: What if the inputs contain Unicode characters? 
```
from collections import defaultdict
    def isAnagram(self, s: str, t: str) -> bool:
        s_map = defaultdict(int)
        t_map = defaultdict(int)
        for c in s:
            s_map[c] += 1
        for c in t:
            t_map[c] += 1

        return s_map == t_map
```
TC: O(n)  SC: O(1)

#### Intersection of Two Arrays
```
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result = set()
        nums1_set = set(nums1)
        for n in nums2:
            if n in nums1_set:
                result.add(n)
        return list(result)
```
TC: O(n+m)  SC: O(1)

#### Happy Number
1. 题目中说了会 无限循环，那么也就是说求和的过程中，sum会重复出现
```
def isHappy(self, n: int) -> bool:
        record = set()
        while True:
            item_list = [int(i)*int(i) for i in str(n)]
            n = sum(item_list)
            if n == 1:
                return True
            if n in record:
                return False
            record.add(n)
```
TC: O(logn)  SC: O(logn)

#### Two Sum
1. 本题，我们不仅要知道元素有没有遍历过，还要知道这个元素对应的下标，需要使用 key value结构来存放，key来存元素，value来存下标，那么使用map正合适
```
def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in hashmap:
                return [hashmap[diff], i]
            else:
                hashmap[nums[i]] = i
```
TC: O(n)  SC: O(n)
