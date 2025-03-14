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
