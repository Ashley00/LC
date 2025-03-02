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
