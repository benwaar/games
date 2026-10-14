mod data;

use wasm_bindgen::prelude::*;
use web_sys::{AudioContext, OscillatorType};

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    Ok(())
}

#[wasm_bindgen]
pub struct AudioPlayer {
    ctx: AudioContext,
    active: Vec<web_sys::OscillatorNode>,
}

#[wasm_bindgen]
impl AudioPlayer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<AudioPlayer, JsValue> {
        let ctx = AudioContext::new()?;
        Ok(AudioPlayer {
            ctx,
            active: Vec::new(),
        })
    }

    pub fn play(&mut self) -> Result<(), JsValue> {
        let _ = self.ctx.resume();
        let dest = self.ctx.destination();
        let now = self.ctx.current_time();
        let mut t = now;

        for i in 0..data::NOTE_COUNT {
            let freq = data::FREQS[i];
            let dur = data::DURATIONS_MS[i] / 1000.0;

            if freq > 0.0 {
                let osc = self.ctx.create_oscillator()?;
                osc.set_type(OscillatorType::Square);
                osc.frequency().set_value(freq as f32);

                let gain = self.ctx.create_gain()?;
                let attack = 0.005_f64.min(dur * 0.1);
                let release = 0.005_f64.min(dur * 0.1);
                gain.gain().set_value_at_time(0.0, t)?;
                gain.gain().linear_ramp_to_value_at_time(0.3, t + attack)?;
                gain.gain().set_value_at_time(0.3, t + dur - release)?;
                gain.gain().linear_ramp_to_value_at_time(0.0, t + dur)?;

                osc.connect_with_audio_node(&gain)?;
                gain.connect_with_audio_node(&dest)?;
                osc.start_with_when(t)?;
                osc.stop_with_when(t + dur)?;
                self.active.push(osc);
            }
            t += dur;
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        for osc in self.active.drain(..) {
            let _ = osc.stop();
        }
    }

    pub fn note_count(&self) -> usize {
        data::NOTE_COUNT
    }
}
