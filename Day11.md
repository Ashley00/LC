#### Day11
#### 150. Evaluate Reverse Polish Notation
栈与递归之间在某种程度上是可以转换的
```
def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        operator = ["+", "-", "*", "/"]

        for c in tokens:
            if c in operator: # do calculation based on operator
                second = int(stack.pop())
                first = int(stack.pop())
                if c == "+":
                    stack.append(first + second)
                elif c == "-":
                    stack.append(first - second)
                elif c == "*":
                    stack.append(first * second)
                else:
                    stack.append(first / second)
            else: # add number to stack
                stack.append(c)
        return int(stack.pop())
```
TC: O(n)  SC: O(n)

#### 239. Sliding Window Maximum
1. 用大顶堆的话要存(value, index)才能移除不在区间范围内的数 O(nlogn)
```
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        heap = []
        output = []
        for i in range(len(nums)):
            heapq.heappush(heap, (-nums[i], i))
            if i >= k - 1:
                while heap[0][1] <= i - k:
                    heapq.heappop(heap)
                output.append(-heap[0][0])
        return output
```
2. Monotonically decreasing queue
队列没有必要维护窗口里的所有元素，只需要维护有可能成为窗口里最大值的元素就可以了，同时保证队列里的元素数值是由大到小的
```
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        result = []
        queue = deque([]) # double sided queue
        left, right = 0, 0
        currMax = max(nums[0:k])
        while right < len(nums):
            if right < k: # initial state
                while len(queue) != 0 and nums[right] > queue[-1]:
                    queue.pop()
                queue.append(nums[right])
            else:
                result.append(queue[0])
                # check left number
                if nums[left] == queue[0]:
                    queue.popleft()
                left += 1
                # check right number
                while len(queue) != 0 and nums[right] > queue[-1]:
                    queue.pop()
                queue.append(nums[right])
            right += 1
        result.append(queue[0]) # last one
        return result
```
TC: O(n)  SC: O(k)

#### 347. Top K Frequent Element
1. Max Heap: frequency then heap
```
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        frequency = {} # count the frequency of each number
        for n in nums:
            if n in frequency:
                frequency[n] += 1
            else:
                frequency[n] = 1
        # max heap to store based on frequency
        heap = [(-freq, val) for val, freq in frequency.items()]
        heapq.heapify(heap)
        result = []
        for i in range(k):
            freq, val = heapq.heappop(heap)
            result.append(val)
        return result
```
TC: O(n+klogn)  SC: O(n+k)

heapify() takes O(n) time for n elements.

Each heappop() operation takes O(log n) time (since the heap size starts at n and shrinks with each pop).

2. Min Heap:
```
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        for num in nums:
            count[num] = 1 + count.get(num, 0)

        heap = []
        for num in count.keys():
            heapq.heappush(heap, (count[num], num))
            if len(heap) > k:
                heapq.heappop(heap)

        res = []
        for i in range(k):
            res.append(heapq.heappop(heap)[1])
        return res
```
TC: O(nlogk)  SC: O(n+k)

We iterate through the count dictionary, which has at most n unique keys.

Each heappush() operation takes O(log k) time because the heap size is at most k.

Each heappop() operation also takes O(log k) time.
