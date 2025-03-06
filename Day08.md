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

