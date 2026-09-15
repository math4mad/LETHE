/**
 * CHORA Apple-Style Enhancement Pack - JavaScript
 * ================================================
 * 功能：
 * 1. IntersectionObserver 滚动入场动画
 * 2. 导航栏滚动收缩
 * 3. 平滑锚点滚动增强
 * 4. 表格横向滚动（移动端触摸优化）
 */

(function() {
  'use strict';

  /**
   * 1. 滚动入场动画 - Apple 风格 fade-up
   * 使用 IntersectionObserver 实现高性能滚动检测
   */
  function initScrollAnimations() {
    const observerOptions = {
      root: null,
      rootMargin: '0px 0px -80px 0px',
      threshold: 0.1
    };

    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          // 动画只触发一次，观察后取消
          observer.unobserve(entry.target);
        }
      });
    }, observerOptions);

    // 观察所有带 .animate-on-scroll 类的元素
    const elements = document.querySelectorAll('.animate-on-scroll');
    elements.forEach(function(el) {
      observer.observe(el);
    });

    console.log('[CHORA Apple] Scroll animations initialized:', elements.length, 'elements');
  }

  /**
   * 2. 导航栏滚动效果 - Apple 玻璃拟态导航栏
   * 滚动时导航栏收紧 + 增强阴影
   */
  function initNavScroll() {
    const nav = document.querySelector('.nav-glass');
    if (!nav) return;

    let lastScrollY = window.scrollY;
    let ticking = false;

    function updateNav() {
      if (window.scrollY > 50) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }
      ticking = false;
    }

    window.addEventListener('scroll', function() {
      if (!ticking) {
        window.requestAnimationFrame(updateNav);
        ticking = true;
      }
    }, { passive: true });

    console.log('[CHORA Apple] Nav scroll effect initialized');
  }

  /**
   * 3. 平滑锚点滚动增强
   * 比 CSS scroll-behavior 更精细的控制
   */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
      anchor.addEventListener('click', function(e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#' || targetId === '#!') return;

        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();

          // 计算导航栏高度偏移
          const navHeight = document.querySelector('.nav-glass')?.offsetHeight || 0;
          const targetPosition = targetElement.getBoundingClientRect().top + window.scrollY - navHeight - 20;

          window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
          });

          // 更新 URL（不滚动）
          history.pushState(null, null, targetId);
        }
      });
    });

    console.log('[CHORA Apple] Smooth scroll initialized');
  }

  /**
   * 4. 表格横向滚动容器（移动端优化）
   * 自动包裹 .apple-table 在可滚动容器中
   */
  function initTableScroll() {
    const tables = document.querySelectorAll('.apple-table');
    const isMobile = window.innerWidth <= 1024;

    tables.forEach(function(table) {
      // 只在移动端添加横向滚动容器
      if (isMobile && !table.parentElement.classList.contains('table-wrapper')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'table-wrapper';
        wrapper.style.overflowX = 'auto';
        wrapper.style.webkitOverflowScrolling = 'touch';
        wrapper.style.margin = '1rem 0';
        wrapper.style.borderRadius = '12px';
        wrapper.style.border = '1px solid rgba(0,0,0,0.08)';

        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });

    // 窗口大小改变时重新评估
    let resizeTimer;
    window.addEventListener('resize', function() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function() {
        // 本仓补丁 (chora@trial, 披露): 原字节为 location.reload() —— 每次跨越 1024px
        // 都会刷新整页,丢弃 fleet 快照与读数。包裹容器只在加载时评估一次;旋转屏幕
        // 不值得一次重载。原钉字节仍在 artifacts/external/ 哈希下可查。
      }, 250);
    });

    console.log('[CHORA Apple] Table scroll initialized');
  }

  /**
   * 5. 按钮点击涟漪效果 - Apple 风格
   */
  function initButtonRipple() {
    document.querySelectorAll('.apple-btn').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        // 创建涟漪元素
        const ripple = document.createElement('span');
        ripple.className = 'btn-ripple';
        const rect = btn.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';

        btn.appendChild(ripple);

        // 动画结束后移除
        setTimeout(function() {
          ripple.remove();
        }, 600);
      });
    });

    // 添加涟漪动画样式
    const style = document.createElement('style');
    style.textContent = `
      .apple-btn { position: relative; overflow: hidden; }
      .btn-ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: btn-ripple-animation 0.6s ease-out;
        pointer-events: none;
      }
      @keyframes btn-ripple-animation {
        to {
          transform: scale(4);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);

    console.log('[CHORA Apple] Button ripple initialized');
  }

  /**
   * 6. 页面加载完成后初始化所有功能
   */
  function init() {
    // 等待 DOM 加载完成
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function() {
        initScrollAnimations();
        initNavScroll();
        initSmoothScroll();
        initTableScroll();
        initButtonRipple();
      });
    } else {
      initScrollAnimations();
      initNavScroll();
      initSmoothScroll();
      initTableScroll();
      initButtonRipple();
    }
  }

  // 启动
  if (typeof document !== 'undefined') {
    init();
  }

})();
