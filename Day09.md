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
