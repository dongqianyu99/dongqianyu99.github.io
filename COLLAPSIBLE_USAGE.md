# 折叠列表功能使用说明

这是你博客的新功能！现在你可以创建可折叠的内容区域，帮助管理长篇内容并改善阅读体验。

## 如何使用

在你的Markdown文件中使用以下语法：

```markdown
{{< collapsible title="你的标题" >}}
这里是可以折叠的内容。

你可以包含：
- 列表
- **加粗文本**
- [链接](https://example.com)
- 数学公式 $E = mc^2$
- 图片
- 任何其他Markdown内容

{{< /collapsible >}}
```

## 示例

### 基本用法
```markdown
{{< collapsible title="点击展开详细内容" >}}
这是一个简单的折叠内容示例。
{{< /collapsible >}}
```

### 包含复杂内容
```markdown
{{< collapsible title="📚 学习资料" >}}
## 视频教程
- [教程1](link1)
- [教程2](link2)

## 参考资料
1. 重要文档
2. 相关论文

### 数学公式
$$E = mc^2$$

{{< /collapsible >}}
```

### 嵌套使用
```markdown
{{< collapsible title="主题目" >}}
这是外层内容。

{{< collapsible title="子主题" >}}
这是内层的折叠内容。
{{< /collapsible >}}

{{< /collapsible >}}
```

## 特性

- ✨ 平滑动画效果
- 🎯 支持键盘导航（回车键和空格键）
- ♿ 符合无障碍访问标准（ARIA）
- 📱 响应式设计
- 🎨 与你的博客主题色彩方案一致

## 已应用的示例

检查你的 `CS285_Lecture_5.md` 文件，我已经将视频链接部分改为了折叠形式作为示例。

现在你可以访问 http://localhost:1313/lee-log/ 来查看效果！