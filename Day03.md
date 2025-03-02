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
