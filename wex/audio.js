export function spatialAudio(ac, masterGain, x, canvasWidth) {
  const panner = ac.createPanner();
  panner.panningModel = 'HRTF';
  panner.distanceModel = 'inverse';
  panner.refDistance = 1;
  panner.maxDistance = 10;
  const pan = (x / canvasWidth - 0.5) * 2;
  panner.positionX.value = pan;
  panner.positionY.value = 0;
  panner.positionZ.value = -1;
  panner.connect(masterGain || ac.destination);
  return panner;
}

export function scheduleAudio(ac, masterGain, audioNotes, canvas, volumeSlider, pauseBtn, spatialAudioFn) {
  if (!audioNotes || !ac || !masterGain) return false;
  masterGain.gain.value = volumeSlider.value / 100;
  const positions = audioNotes.positions;
  const canvasWidth = canvas.width || 256;
  let t = ac.currentTime + 0.05;
  for (let i = 0; i < audioNotes.freqs.length; i++) {
    const freq = audioNotes.freqs[i];
    const dur  = audioNotes.durs[i] / 1000;
    if (freq > 0) {
      const osc = ac.createOscillator();
      osc.type = 'square';
      osc.frequency.value = freq;
      if (positions && positions[i]) {
        const panner = spatialAudioFn(ac, masterGain, positions[i].x, canvasWidth);
        osc.connect(panner);
      } else {
        osc.connect(masterGain);
      }
      osc.start(t);
      osc.stop(t + dur);
    }
    t += dur;
  }
  pauseBtn.textContent = '⏸ Pause (P)';
  return true;
}

export function togglePause(ac, playing, pauseBtn) {
  if (!ac) return playing;
  if (ac.state === 'running') {
    ac.suspend();
    pauseBtn.textContent = '▶ Play (P)';
    return false;
  } else {
    ac.resume();
    pauseBtn.textContent = '⏸ Pause (P)';
    return true;
  }
}

export async function startAudio(audioNotes, overlay, scheduleAudioFn) {
  const ac = new AudioContext();
  const masterGain = ac.createGain();
  masterGain.connect(ac.destination);
  if (ac.state === 'suspended') {
    overlay.style.display = 'flex';
    await new Promise(resolve => {
      overlay.addEventListener('click', async () => {
        await ac.resume();
        overlay.style.display = 'none';
        resolve();
      }, { once: true });
    });
  }
  scheduleAudioFn(ac, masterGain);
  return { ac, masterGain };
}
