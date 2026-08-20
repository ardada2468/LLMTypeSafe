/*
 * Drives the hero validator: rows arrive, each is checked against its declared
 * type in turn, and the one that fails throws. Loops while on screen.
 *
 * Reduced motion gets the finished state immediately and no loop — the point
 * of the figure is the failure, not the animation.
 */
(function () {
  var fig = document.getElementById('validator');
  var thrown = document.getElementById('throw');
  if (!fig || !thrown) return;

  var rows = Array.prototype.slice.call(fig.querySelectorAll('.row'));
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  var timers = [];
  var running = false;

  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function at(ms, fn) {
    timers.push(setTimeout(fn, ms));
  }

  function reset() {
    rows.forEach(function (r) { r.classList.remove('in', 'checked'); });
    thrown.classList.remove('on');
  }

  function settle() {
    rows.forEach(function (r) { r.classList.add('in', 'checked'); });
    thrown.classList.add('on');
  }

  function play() {
    if (running) return;
    running = true;
    reset();

    var t = 260;
    rows.forEach(function (row) {
      at(t, function () { row.classList.add('in'); });
      t += 190;
    });

    t += 320;
    rows.forEach(function (row) {
      at(t, function () { row.classList.add('checked'); });
      t += 620;
    });

    at(t + 120, function () { thrown.classList.add('on'); });

    // hold the failed state long enough to read, then run it again
    at(t + 4200, function () {
      running = false;
      play();
    });
  }

  function stop() {
    clearTimers();
    running = false;
  }

  function start() {
    if (reduced.matches) { clearTimers(); settle(); return; }
    play();
  }

  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) start();
      else stop();
    });
  }, { threshold: 0.25 });

  seen.observe(fig);

  // honour a mid-session change to the motion preference
  var onPref = function () { stop(); start(); };
  if (reduced.addEventListener) reduced.addEventListener('change', onPref);
  else if (reduced.addListener) reduced.addListener(onPref);

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop();
    else if (!reduced.matches) start();
  });
})();
