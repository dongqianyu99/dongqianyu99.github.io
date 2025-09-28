// 折叠列表功能
document.addEventListener('DOMContentLoaded', function() {
    const collapsibles = document.querySelectorAll('.collapsible');

    collapsibles.forEach(function(collapsible) {
        collapsible.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;

            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('active');

                // 平滑动画效果
                if (content.classList.contains('active')) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                } else {
                    content.style.maxHeight = '0';
                }
            }
        });
    });

    // 支持键盘访问
    collapsibles.forEach(function(collapsible) {
        collapsible.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });

        // 添加适当的ARIA属性
        collapsible.setAttribute('role', 'button');
        collapsible.setAttribute('aria-expanded', 'false');

        const content = collapsible.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
            const contentId = 'collapsible-content-' + Math.random().toString(36).substr(2, 9);
            content.setAttribute('id', contentId);
            collapsible.setAttribute('aria-controls', contentId);

            // 监听状态变化来更新ARIA属性
            const observer = new MutationObserver(function() {
                const isActive = collapsible.classList.contains('active');
                collapsible.setAttribute('aria-expanded', isActive ? 'true' : 'false');
            });

            observer.observe(collapsible, { attributes: true, attributeFilter: ['class'] });
        }
    });
});