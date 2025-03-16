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
