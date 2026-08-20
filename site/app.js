/*
 * Drives the hero rig.
 *
 * It runs two takes, alternating, because one alone only tells half the story:
 *
 *   take 1  a well-formed reply  -> three typed values land in your hands
 *   take 2  the same task, but the model answers "high" where a number was
 *           declared -> ValidationError instead of a string wearing the wrong
 *           type
 *
 * Showing only the failure would read as "this library reports errors". Showing
 * only the success would read as "this library parses text". The pair is the
 * actual proposition: you get typed values, and when you cannot, you are told.
 */
(function () {
  var rig = document.getElementById('rig');
  if (!rig) return;

  var rowsEl = document.getElementById('rigRows');
  var scan = document.getElementById('scan');
  var stateEl = document.getElementById('rigState');
  var footEl = document.getElementById('rigFoot');
  var rows = [].slice.call(rig.querySelectorAll('.rrow'));

  var TAKES = [
    {
      ok: true,
      reply: ['answer: Paris', 'score: 0.92', 'sources: wiki, britannica'],
      value: [
        '<span class="q">"</span>Paris<span class="q">"</span>',
        '<span class="n">0.92</span>',
        '<span class="q">[</span>"wiki", "britannica"<span class="q">]</span>',
      ],
      bad: -1,
      tag: 'Prediction',
      msg: 'three fields, each the type you declared',
    },
    {
      ok: false,
      reply: ['answer: Paris', 'score: high', 'sources: wiki, britannica'],
      value: [
        '<span class="q">"</span>Paris<span class="q">"</span>',
        'expected number',
        '<span class="q">[</span>"wiki", "britannica"<span class="q">]</span>',
      ],
      bad: 1,
      tag: 'ValidationError',
      msg: 'score: expected number, received "high"',
    },
  ];

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  var timers = [];
  var running = false;
  var take = 0;

  function clear() { timers.forEach(clearTimeout); timers = []; }
  function at(ms, fn) { timers.push(setTimeout(fn, ms)); }
  function say(s) { if (stateEl) stateEl.textContent = s; }

  function reset() {
    rig.classList.remove('live', 'ok', 'bad');
    scan.classList.remove('sweep');
    footEl.classList.remove('on', 'bad');
    footEl.innerHTML = '';
    rows.forEach(function (r) {
      r.classList.remove('done', 'good', 'bad');
      r.querySelector('.rout').textContent = '';
      r.querySelector('.rout').classList.remove('typing');
      r.querySelector('.rval').innerHTML = '';
    });
    say('idle');
  }

  function fill(t) {
    rows.forEach(function (r, i) {
      r.querySelector('.rout').textContent = t.reply[i];
      r.querySelector('.rval').innerHTML = t.value[i];
      r.classList.add('done', i === t.bad ? 'bad' : 'good');
    });
    footEl.innerHTML = '<span class="foot-tag">' + t.tag + '</span><span class="foot-msg">' + t.msg + '</span>';
    footEl.classList.add('on');
    footEl.classList.toggle('bad', !t.ok);
    rig.classList.add(t.ok ? 'ok' : 'bad');
    say(t.ok ? 'valid' : '1 invalid');
  }

  function settle() { clear(); reset(); fill(TAKES[1]); }

  function type(t, i, done) {
    if (i >= rows.length) { done(); return; }
    var el = rows[i].querySelector('.rout');
    var text = t.reply[i];
    var n = 0;
    el.classList.add('typing');
    (function tick() {
      n = Math.min(text.length, n + 2);
      el.textContent = text.slice(0, n);
      if (n < text.length) at(17, tick);
      else { el.classList.remove('typing'); at(85, function () { type(t, i + 1, done); }); }
    })();
  }

  function play() {
    if (running) return;
    running = true;

    var t = TAKES[take % TAKES.length];
    take++;
    reset();

    at(300, function () {
      rig.classList.add('live');
      say('streaming');

      type(t, 0, function () {
        say('validating');
        scan.style.setProperty('--sweep-to', rowsEl.getBoundingClientRect().height + 'px');
        scan.classList.remove('sweep');
        void scan.offsetWidth;
        scan.classList.add('sweep');

        rows.forEach(function (row, i) {
          at(230 + i * 400, function () {
            // The value column is the payoff — it lands as the sweep passes.
            row.querySelector('.rval').innerHTML = t.value[i];
            row.classList.add('done', i === t.bad ? 'bad' : 'good');
          });
        });

        var end = 230 + rows.length * 400 + 110;
        at(end, function () {
          rig.classList.remove('live');
          rig.classList.add(t.ok ? 'ok' : 'bad');
          footEl.innerHTML = '<span class="foot-tag">' + t.tag + '</span><span class="foot-msg">' + t.msg + '</span>';
          footEl.classList.add('on');
          footEl.classList.toggle('bad', !t.ok);
          say(t.ok ? 'valid' : '1 invalid');
        });

        at(end + 3800, function () { running = false; play(); });
      });
    });
  }

  function stop() { clear(); running = false; }
  function start() { if (reduced.matches) settle(); else play(); }

  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) start(); else stop(); });
  }, { threshold: 0.3 });
  seen.observe(rig);

  var onPref = function () { stop(); start(); };
  if (reduced.addEventListener) reduced.addEventListener('change', onPref);
  else if (reduced.addListener) reduced.addListener(onPref);

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop(); else if (!reduced.matches) { stop(); play(); }
  });
})();
