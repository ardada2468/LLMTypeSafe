/*
 * Diagram + docs behaviour.
 *
 * Figures animate once when scrolled into view: nodes settle in, edges draw
 * themselves along their own measured length, labels follow. Edge length is
 * measured from the DOM rather than guessed, so the dash timing is right
 * whatever the path.
 *
 * Reduced motion is handled in CSS — everything is simply already drawn — so
 * this file only needs to avoid staging work that would fight it.
 */
(function () {
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* ── figures ─────────────────────────────────────────── */

  var figs = [].slice.call(document.querySelectorAll('.fig[data-anim]'));

  figs.forEach(function (fig) {
    // Give every edge its own length so the draw reads evenly.
    [].slice.call(fig.querySelectorAll('.g-edge')).forEach(function (edge) {
      var len;
      try {
        len = Math.ceil(edge.getTotalLength());
      } catch (e) {
        len = 400;
      }
      edge.style.setProperty('--len', len);
    });

    // Stagger by document order: nodes, then edges, then labels.
    var order = [].slice.call(fig.querySelectorAll('.g-node, .g-edge, .g-lbl'));
    order.forEach(function (el, i) {
      el.style.transitionDelay = reduced.matches ? '0s' : (i * 55) + 'ms';
    });
  });

  if (figs.length) {
    var figSeen = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('run');
        figSeen.unobserve(entry.target); // draw once; it is a diagram, not a loop
      });
    }, { threshold: 0.18, rootMargin: '0px 0px -8% 0px' });

    figs.forEach(function (f) { figSeen.observe(f); });
  }

  /* ── table of contents highlighting ──────────────────── */

  var links = [].slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  if (!links.length) return;

  var byId = {};
  var targets = [];
  links.forEach(function (a) {
    var el = document.getElementById(a.getAttribute('href').slice(1));
    if (!el) return;
    byId[el.id] = a;
    targets.push(el);
  });

  function mark(id) {
    links.forEach(function (a) { a.classList.remove('here'); });
    if (byId[id]) byId[id].classList.add('here');
  }

  var secSeen = new IntersectionObserver(function (entries) {
    // Pick the entry nearest the top of the viewport that is on screen.
    var best = null;
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      if (!best || e.boundingClientRect.top < best.boundingClientRect.top) best = e;
    });
    if (best) mark(best.target.id);
  }, { rootMargin: '-88px 0px -62% 0px', threshold: 0 });

  targets.forEach(function (t) { secSeen.observe(t); });
})();
