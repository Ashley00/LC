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

