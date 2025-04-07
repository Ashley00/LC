#### Day14
#### 226. Invert Binary Tree
想要翻转它，其实就把每一个节点的左右孩子交换一下就可以了
```
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root == None:
            return None
        left = root.left
        root.left = root.right
        root.right = left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
        
```

#### 101. Symmetric Tree
1. 要比较的是两个树（这两个树是根节点的左右子树)
2. 要遍历两棵树而且要比较内侧和外侧节点，所以准确的来说是一个树的遍历顺序是左右中，一个树的遍历顺序是右左中
3. Recursion. Iterative: using queue
```
    def checkChildSymmetric(self, left, right):
        # 4 different conditions for left and right
        if left == None and right == None:
            return True
        elif left == None and right:
            return False
        elif left and right == None:
            return False
        else: # left and right not None
            # check left value and right value
            if left.val == right.val:
                outer = self.checkChildSymmetric(left.left, right.right)
                inner = self.checkChildSymmetric(left.right, right.left)
                return outer and inner
            else:
                # left and right value different, directly return False
                return False

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root == None: # check root first
            return False
        return self.checkChildSymmetric(root.left, root.right)
        
```

#### 104. Maximum depth of binary tree
Iterative:
```
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        queue = deque([])
        queue.append(root)
        result = 0
        while len(queue) != 0:
            length = len(queue)
            result += 1
            for i in range(length):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result
```
Recursion:
```
    def maxdepth(self, root: treenode) -> int:
        return self.getdepth(root)
        
    def getdepth(self, node):
        if not node:
            return 0
        leftheight = self.getdepth(node.left) #左
        rightheight = self.getdepth(node.right) #右
        height = 1 + max(leftheight, rightheight) #中
        return height
```

#### 111. Minimum Depth of Binary Tree
Recursion:
```
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        queue = deque([])
        queue.append(root)
        result = 0
        while len(queue) != 0:
            length = len(queue)
            result += 1
            for i in range(length):
                node = queue.popleft()
                if node.left == None and node.right == None:
                    return result
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
```
Iterative:
```
    def getDepth(self, node):
        if node is None:
            return 0
        leftDepth = self.getDepth(node.left)  # 左
        rightDepth = self.getDepth(node.right)  # 右
        
        # 当一个左子树为空，右不为空，这时并不是最低点
        if node.left is None and node.right is not None:
            return 1 + rightDepth
        
        # 当一个右子树为空，左不为空，这时并不是最低点
        if node.left is not None and node.right is None:
            return 1 + leftDepth
        
        result = 1 + min(leftDepth, rightDepth)
        return result

    def minDepth(self, root):
        return self.getDepth(root)

```
