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

#### Day03
#### 链表基础
1. 类型：单链表，双链表，循环列表
2. 存储方式：链表在内存中可不是连续分布的，分配机制取决于操作系统的内存管理
3. 定义： 
   ```
   class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
   ```
<img src="https://github.com/user-attachments/assets/0ad17bf2-7948-4730-9c4e-eedf35b27176" width="500">

#### 203. Remove Linked List Elements
1. 设置一个虚拟头结点在进行删除操作
```
def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if head is None:
            return None
        
        dummyHead = ListNode(val=0, next=head)
        prev = dummyHead
        curr = dummyHead.next
        
        while curr != None:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummyHead.next
```
TC: O(n)  SC: O(1)

#### Design Linked List
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        curr = self.dummy.next
        for i in range(index):
            curr = curr.next
        return curr.val

    def addAtHead(self, val: int) -> None:
        nextNode = self.dummy.next
        self.dummy.next = ListNode(val, next=nextNode)
        self.size += 1
        
    def addAtTail(self, val: int) -> None:
        curr = self.dummy
        for i in range(self.size):
            curr = curr.next
        curr.next = ListNode(val, next=None)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        curr = self.dummy
        for i in range(index):
            curr = curr.next
        nextNode = curr.next
        curr.next = ListNode(val, next=nextNode)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        curr = self.dummy
        for i in range(index):
            curr = curr.next
        curr.next = curr.next.next
        self.size -= 1
```

#### 206. Reverse Linked List
1. Two pointer: 只需要改变链表的next指针的指向，直接将链表反转 ，而不用重新定义一个新的链表
```
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            temp = curr.next # keep a record of curr.next
            curr.next = prev # reverse
            prev = curr
            curr = temp
        return prev
```
TC: O(n)  SC: O(1)

#### Day04
#### 24. Swap Nodes in Pairs
1. next node的转换
```
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(val=0, next=head)
        prev = dummy
        curr = dummy.next
        while curr and curr.next:
            temp = curr.next.next
            # update next
            prev.next = curr.next
            curr.next.next = curr
            curr.next = temp
            # update prev and curr
            prev = curr
            curr = temp
        return dummy.next
```
TC: O(n)  SC: O(1)

#### 19. Remove Nth Node From End of List
1. Two pointer: fast-slow pointer to find the prev pointer position
2. 删除第N个节点，那么我们当前遍历的指针一定要指向 第N个节点的前一个节点
```
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        prev = dummy
        curr = dummy
        # move the fast pointer to be n step ahead
        for i in range(n):
            curr = curr.next
        # find the prev pointer position
        while curr.next:
            curr = curr.next
            prev = prev.next
        prev.next = prev.next.next
        return dummy.next
```
TC: O(n)  SC: O(1)

#### 160. Intersection of Two Linked Lists
1. 交点不是数值相等，而是指针相等
2. 求出两个链表的长度，并求出两个链表长度的差值，然后让curA移动到，和curB对齐的位置
```
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        # find size of A and B
        sizeA, sizeB = 0, 0
        currA, currB = headA, headB
        while currA:
            currA = currA.next
            sizeA += 1
        while currB:
            currB = currB.next
            sizeB += 1

        # point to the same starting node
        currA, currB = headA, headB
        if sizeA > sizeB:
            for i in range(sizeA-sizeB):
                currA = currA.next
        else:
            for i in range(sizeB-sizeA):
                currB = currB.next
        
        # find intersection
        while currA and currB:
            if currA == currB:
                return currA
            currA = currA.next
            currB = currB.next
        return None
```
TC: O(n+m)  SC: O(1)

#### Linked List Cycle II
1. extra memory to create a set() - can be improved
```
def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        hashmap = set()
        curr = head
        while curr:
            if curr in hashmap:
                return curr
            hashmap.add(curr)
            curr = curr.next
        return None
```
2. 快慢指针法，分别定义 fast 和 slow 指针，从头结点出发，fast指针每次移动两个节点，slow指针每次移动一个节点.
   如果 fast 和 slow指针在途中相遇 ，说明这个链表有环
![image](https://github.com/user-attachments/assets/77d84e9d-f55b-43b3-a9ed-b62eb1b9feb2)

那么相遇时： slow指针走过的节点数为: x + y， fast指针走过的节点数：x + y + n (y + z)，n为fast指针在环内走了n圈才遇到slow指针， （y+z）为 一圈内节点的个数A。

因为fast指针是一步走两个节点，slow指针一步走一个节点， 所以 fast指针走过的节点数 = slow指针走过的节点数 * 2：

(x + y) * 2 = x + y + n (y + z)

x = (n - 1) (y + z) + z

当 n为1的时候，公式就化解为 x = z，

这就意味着，从头结点出发一个指针，从相遇节点 也出发一个指针，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是 环形入口的节点。
```
def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # cycle found
            if slow == fast:
                start1, start2 = head, slow
                while start1 and start2:
                    if start1 == start2: # cycle entrance found
                        return start1
                    start1 = start1.next
                    start2 = start2.next
        return None
```
TC: O(n)  SC: O(1)

![image](https://github.com/user-attachments/assets/aba5103f-663c-4cac-9196-12f02f598f1e)

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


#### Day08

#### 344. Reverse String
```
# Two pointer
def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        n = len(s)
        for i in range(n // 2):
            temp = s[i]
            s[i] = s[n-i-1]
            s[n-i-1] = temp
```
```
# Stack
def reverseString(self, s: List[str]) -> None:
        stack = []
        for char in s:
            stack.append(char)
        for i in range(len(s)):
            s[i] = stack.pop()
```
TC: O(n)  SC: O(1)

#### 541 Reverse String II
其实在遍历字符串的过程中，只要让 i += (2 * k)，i 每次移动 2 * k 就可以了，然后判断是否需要有反转的区间。
```
# helper function to reverse a string list, return a list
    def reverse(self, s_list):
        n = len(s_list)
        for i in range(n // 2):
            temp = s_list[i]
            s_list[i] = s_list[n-i-1]
            s_list[n-i-1] = temp
        return s_list

    def reverseStr(self, s: str, k: int) -> str:
        s_list = list(s) # convert to list
        length = len(s)
        result = [] # keep a result as list
        i = 0
        while i < length:
            if i+2*k <= length: # reverse first k, keep second k
                result = result + self.reverse(s_list[i:i+k]) + s_list[i+k:i+2*k]
                i += 2*k
            else:
                if length-i < k: # fewer than k chars, reverse all
                    result = result + self.reverse(s_list[i:length])
                else: # greater or equal to k, but less than 2k
                    result = result + self.reverse(s_list[i:i+k]) + s_list[i+k:length]
                break
        return ''.join(result) # convert back to string

  def reverseStr(self, s: str, k: int) -> str:
        ls = list(s) # convert str to list
        for i in range(0, len(ls), 2*k):
            if i+k <= len(ls):
                ls[i:i+k] = self.reverse_substring(ls[i:i+k])
            else:
                ls[i:len(ls)] = self.reverse_substring(ls[i:len(ls)])

        return ''.join(ls)
```
TC: O(n)  SC: O(n)

#### Substitute Number
Two Pointer:
```
def substituteNumber(self, s: List[str]) -> str:
        length = 0 # calculate new length
        for c in s:
            if c.isdigit():
                length += 6
            else:
                length += 1

        result = [0] * length
        result[0:len(s)] = s
        oldPointer = len(s) - 1 # end of original string
        newPointer = length - 1
        number = 'rebmun'
        while oldPointer >= 0:
            if oldPointer.isdigit():
                for c in number:
                    result[newPointer] = c
                    newPointer -= 1
            else:
                result[newPointer] = result[oldPointer]
                newPointer -= 1
            oldPointer -= 1
        return result
``
TC: O(n)  SC: O(1)

#### Day09
#### 151. Reverse Words in a String
split()函数能够自动忽略多余的空白字符, default separator is any whitespace
Option1: 使用双指针
Option2: 遇到空格就说明前面的是一个单词，把它加入到一个数组中
Option3: use split(), then swap
```
def reverseWords(self, s: str) -> str:
        s_list = s.split() # default is any whitespace
        n = len(s_list)
        # swap by word
        for i in range(n // 2):
            temp = s_list[i].strip()
            s_list[i] = s_list[n-i-1].strip()
            s_list[n-i-1] = temp
        

        return " ".join(s_list)
```

#### Right rotate String
字符串的右旋转操作是把字符串尾部的若干个字符转移到字符串的前面。给定一个字符串 s 和一个正整数 k，请编写一个函数，将字符串中的后面 k 个字符移到字符串的前面，实现字符串的右旋转操作。
```
    def reverse(self, s_list):
        for i in range(n // 2):
            temp = s_list[i]
            s_list[i] = s_list[n-i-1]
            s_list[n-i-1] = temp
        return s_list

    def rightRotate(self, s: str, k: int) -> str:
        s_list = list(s)
        n = len(s_list)
        # reverse whole string
        s_list = self.reverse(s_list)
        # reverse two parts
        s_list[0:k] = self.reverse(s_list[0:k])
        s_list[k:n] = self.reverse(s_list[k:n])
        
        return ''.join(s_list)
```

#### Day10
#### 栈与队列理论基础
队列是先进先出，栈是先进后出

#### 232. Implement Queue using Stacks
Two stacks to implement.
```
class MyQueue:
    from collections import deque
    def __init__(self):
        self.stack_in = deque([])
        self.stack_out = deque([])
        
    def push(self, x: int) -> None:
        self.stack_in.append(x)

    def pop(self) -> int:
        while len(self.stack_in) != 0:
            self.stack_out.append(self.stack_in.pop())
        poped = self.stack_out.pop()
        while len(self.stack_out) != 0:
            self.stack_in.append(self.stack_out.pop())
        return poped

    def peek(self) -> int:
        while len(self.stack_in) != 0:
            self.stack_out.append(self.stack_in.pop())
        result = self.stack_out[-1]
        while len(self.stack_out) != 0:
            self.stack_in.append(self.stack_out.pop())
        return result
        
    def empty(self) -> bool:
        return len(self.stack_in) == 0
```
Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity?
但在pop的时候，操作就复杂一些，输出栈如果为空，就把进栈数据全部导入进来（注意是全部导入）
如果进栈和出栈都为空的话，说明模拟的队列为空了
```
    def push(self, x: int) -> None:
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None
        
        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans

    def empty(self) -> bool:
        if len(self.stack_in) != 0 or len(self.stack_out) != 0:
            return False
        return True
```
TC: all O(1)  SC: O(n)

#### 225. Implement Stack using Queue
用两个队列que1和que2实现队列的功能，que2其实完全就是一个备份的作用
优化： 一个队列在模拟栈弹出元素的时候只要将队列头部的元素（除了最后一个元素外） 重新添加到队列尾部，此时再去弹出元素就是栈的顺序了
```
class MyStack:
    from collections import deque
    def __init__(self):
        self.queue = deque([])

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        if len(self.queue) == 0:
            return None
        for i in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
        return self.queue.popleft()

    def top(self) -> int:
        poped = self.pop()
        self.queue.append(poped)
        return poped
        
    def empty(self) -> bool:
        return len(self.queue) == 0
```
TC： pop-O(n) top-O(n) SC: O(n)

#### 20. Valid Parentheses
括号匹配是使用栈解决的经典问题
```
from collections import deque
    def isValid(self, s: str) -> bool:
        stack = deque([])
        for c in s:
            if c == "(" or c == "{" or c == "[":
                stack.append(c)
            else:
                if len(stack) == 0:
                    return False
                popedC = stack.pop()
                if c == ")" and popedC == "(":
                    continue
                if c == "]" and  popedC == "[":
                    continue
                if c == "}" and popedC == "{":
                    continue
                return False
        if len(stack) != 0:
            return False
        return True
```
TC: O(n)  SC: O(n)

#### 1047. Remove All Adjacent Duplicates in String
```
  def removeDuplicates(self, s: str) -> str:
        stack = []
        for c in s:
            if len(stack) != 0:
                if c == stack[-1]:
                    stack.pop()
                else:
                    stack.append(c)
            else:
                stack.append(c)
        return ''.join(stack)
```
TC: O(n)  SC: O(n)

