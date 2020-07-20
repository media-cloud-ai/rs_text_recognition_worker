#[macro_use]
extern crate serde_derive;

use mcai_worker_sdk::{
  start_worker,
  trace,
  FormatContext,
  Frame,
  JsonSchema,
  MessageError,
  MessageEvent,
  ProcessResult,
  Version
};

use stainless_ffmpeg_sys::{
  av_get_bits_per_pixel, av_pix_fmt_desc_get,
  AVPixelFormat,
};

use std::sync::{Arc, Mutex};

pub mod built_info {
  include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

#[derive(Debug, Serialize)]
pub struct RecognisedText {
  pts: u64,
  text: String,
}

#[derive(Debug, Default)]
struct TextRecognitionEvent {
  language: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WorkerParameters {
  source_path: String,
  destination_path: String,
  language: Option<String>
}

impl MessageEvent<WorkerParameters> for TextRecognitionEvent {
  fn get_name(&self) -> String {
    "Text recognition".to_string()
  }

  fn get_short_description(&self) -> String {
    "Text recognition worker".to_string()
  }

  fn get_description(&self) -> String {
    r#"This worker applies OCR algorithm on the frame specified as parameter.
It returns the detected text for each requested frame."#
      .to_string()
  }

  fn get_version(&self) -> Version {
    Version::parse(built_info::PKG_VERSION).expect("unable to locate Package version")
  }

  fn init_process(&mut self, parameters: WorkerParameters, _format_context: Arc<Mutex<FormatContext>>) -> Result<Vec<usize>, MessageError> {
    self.language = parameters.language.unwrap_or_else(|| "eng".to_string());

    Ok(vec![0])
  }

  fn process_frame(
    &mut self,
    job_id: &str,
    _stream_index: usize,
    frame: Frame,
  ) -> Result<ProcessResult, MessageError> {

    let recognised_text =
      unsafe {
        let pixel_format = std::mem::transmute::<_, AVPixelFormat>((*frame.frame).format);

        let av_pix_fmt_desc = av_pix_fmt_desc_get(pixel_format);
        let bytes_per_pixel = av_get_bits_per_pixel(av_pix_fmt_desc) / 8;

        let width = (*frame.frame).width;
        let height = (*frame.frame).height;
        let linesize = (*frame.frame).linesize[0];

        let buffer_size = (linesize * height) as usize;

        let data: Vec<u8> = Vec::from_raw_parts(
          (*frame.frame).data[0],
          buffer_size,
          buffer_size,
        );

        let text = tesseract::ocr_from_frame(
          &data,
          width,
          height,
          bytes_per_pixel,
          linesize,
          &self.language,
        ).unwrap();
        trace!(target: &job_id, "{:?}", text);

        std::mem::forget(data);

        RecognisedText {
          pts: (*frame.frame).pts as u64,
          text,
        }
      };

    Ok(ProcessResult::new_json(recognised_text))
  }
}

fn main() {
  let worker = TextRecognitionEvent::default();
  start_worker(worker);
}
