export function speak(text) {
  if (!window.speechSynthesis || !text) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

export function buildMirrorDom(mirror, canvas, nodes, cw, ch, voiceEnabled, speak) {
  mirror.innerHTML = '';
  const announcer = document.createElement('div');
  announcer.setAttribute('aria-live', 'polite');
  announcer.className = 'sr-only';
  announcer.id = 'announcer';
  mirror.appendChild(announcer);

  for (const node of nodes) {
    const div = document.createElement('div');
    div.setAttribute('role', node.role || 'region');
    div.setAttribute('aria-label', node.label || '');
    div.tabIndex = 0;
    div.style.left = ((node.x || 0) / cw * 100) + '%';
    div.style.top = ((node.y || 0) / ch * 100) + '%';
    div.style.width = ((node.w || cw) / cw * 100) + '%';
    div.style.height = ((node.h || ch) / ch * 100) + '%';
    div.className = 'sr-only';
    div.addEventListener('focus', () => { if (voiceEnabled) speak(node.label || ''); });
    mirror.appendChild(div);
  }

  if (nodes.length > 0) {
    canvas.setAttribute('aria-label', nodes[0].label || 'Game content');
  }
}
