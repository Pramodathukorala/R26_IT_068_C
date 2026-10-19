/**
 * HTML for a hidden 1px WebView that loops an MP3 alarm on demand.
 *
 * The page exposes two globals named by the caller (e.g. startAlarm/stopAlarm),
 * which RN drives with injectJavaScript. The WebView must be rendered with
 * mediaPlaybackRequiresUserAction={false} so play() works without a tap.
 *
 * @param {string} soundUri  data: URI of the MP3 (see src/assets/sounds)
 * @param {string} startName name of the global that starts the loop
 * @param {string} stopName  name of the global that stops it
 */
export function alarmSoundHtml(soundUri, startName, stopName) {
  return `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
<audio id="alarm" src="${soundUri}" loop preload="auto"></audio>
<script>
  var el = document.getElementById('alarm');
  var playing = false;

  function ${startName}() {
    if (playing) { return; }
    playing = true;
    el.currentTime = 0;
    var p = el.play();
    if (p && p.catch) { p.catch(function () { playing = false; }); }
  }

  function ${stopName}() {
    playing = false;
    el.pause();
    el.currentTime = 0;
  }
</script></body></html>`;
}
