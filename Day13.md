#### Day13
#### 二叉树理论基础 Binary Tree
1. 满二叉树：如果一棵二叉树只有度为0的结点和度为2的结点，并且度为0的结点在同一层上，则这棵二叉树为满二叉树.深度为k，有2^k-1个节点
2. 完全二叉树：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。
   若最底层为第 h 层（h从1开始），则该层包含 1~ 2^(h-1) 个节点。
3. 二叉搜索树: 一个有序树
4. 二叉树的遍历方式: 深度优先遍历：先往深走，遇到叶子节点再往回走。广度优先遍历：一层一层的去遍历。这里前中后，其实指的就是中间节点的遍历顺序

#### 二叉树的递归遍历
每次写递归，都按照这三要素来写:
确定递归函数的参数和返回值, 确定终止条件, 确定单层递归的逻辑
144. Binary Tree Preorder Traversal
```
    def dfs(self, root, result):
        if root == None: # base case
            return
        # each recursion logic
        result.append(root.val)
        self.dfs(root.left, result)
        self.dfs(root.right, result)

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        self.dfs(root, result) # define parameter and return type
        return result
```

#### 二叉树的统一迭代法
加一个 boolean 值跟随每个节点，false (默认值) 表示需要为该节点和它的左右儿子安排在栈中的位次，true 表示该节点的位次之前已经安排过了，可以收割节点了.
```
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        result = []
        if root != None:
            stack.append((root, False))
        while len(stack) != 0:
            node, visited = stack.pop()
            if visited: # check if this node is been processed before
                result.append(node.val)
                continue
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
            stack.append((node, True))
        return result
```

#### 102. Binary Tree Level Order Traversal
1. 队列先进先出，符合一层一层遍历的逻辑，而用栈先进后出适合模拟深度优先遍历也就是递归的逻辑。
2. 这种层序遍历方式就是图论中的广度优先遍历，只不过我们应用在二叉树上.
```
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root == None:
            return []
        result = []
        queue = deque([root])
        while len(queue) != 0:
            n = len(queue)
            level = []
            for _ in range(n):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
```
