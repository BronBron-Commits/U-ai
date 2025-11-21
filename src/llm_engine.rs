use anyhow::Result;
use llm::{InferenceSession, InferenceSessionConfig, InferenceParameters, Model};

pub struct Engine<'m> {
    pub model: &'m dyn Model,
    pub session: InferenceSession,
}

impl<'m> Engine<'m> {
    pub fn new(model: &'m dyn Model) -> Self {
        let session = model.start_session(InferenceSessionConfig::default());
        Self { model, session }
    }

    pub fn infer(&mut self, prompt: &str) -> Result<String> {
        let mut output = String::new();

        self.session.infer(
            self.model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: prompt.into(),
                parameters: InferenceParameters::default(),
                play_back_previous_tokens: false,
                ..Default::default()
            },
            &mut llm::OutputRequest::new(Some(&mut output), None),
        )?;

        Ok(output)
    }
}
