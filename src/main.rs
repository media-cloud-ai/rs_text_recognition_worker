use mcai_worker_sdk::job::{Job, JobResult};
use mcai_worker_sdk::start_worker;
use mcai_worker_sdk::worker::{Parameter, ParameterType};
use mcai_worker_sdk::{McaiChannel, MessageError, MessageEvent, Version};

mod message;

macro_rules! crate_version {
  () => {
    env!("CARGO_PKG_VERSION")
  };
}

#[derive(Debug)]
struct TextRecognitionEvent {}

impl MessageEvent for TextRecognitionEvent {
  fn get_name(&self) -> String {
    "Text recognition".to_string()
  }

  fn get_short_description(&self) -> String {
    "Text recognition worker".to_string()
  }

  fn get_description(&self) -> String {
    r#"This worker applies OCR algorithm on the frame specified as parameter.
It returns the detected text as a JSON array."#
      .to_string()
  }

  fn get_version(&self) -> Version {
    Version::parse(crate_version!()).expect("unable to locate Package version")
  }

  fn get_parameters(&self) -> Vec<Parameter> {
    vec![
      Parameter {
        identifier: message::SOURCE_PATH_PARAMETER.to_string(),
        label: "Source path".to_string(),
        kind: vec![ParameterType::String],
        required: true,
      },
      Parameter {
        identifier: message::LANGUAGE_PARAMETER.to_string(),
        label: "The language to be detected".to_string(),
        kind: vec![ParameterType::Integer],
        required: true,
      },
    ]
  }

  fn process(
    &self,
    channel: Option<McaiChannel>,
    job: &Job,
    job_result: JobResult,
  ) -> Result<JobResult, MessageError> {
    message::process(channel, job, job_result)
  }
}

static TEXT_RECOGNITION_EVENT: TextRecognitionEvent = TextRecognitionEvent {};

fn main() {
  start_worker(&TEXT_RECOGNITION_EVENT);
}
