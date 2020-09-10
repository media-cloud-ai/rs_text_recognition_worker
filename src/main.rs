#[macro_use]
extern crate serde_derive;

use mcai_worker_sdk::{
  start_worker, trace, FormatContext, Frame, JsonSchema, MessageError, MessageEvent, ProcessResult,
  RegionOfInterest, Scaling, StreamDescriptor, Version, VideoFilter, VideoFormat,
};

use stainless_ffmpeg_sys::{
  av_get_bits_per_pixel, av_pix_fmt_desc_get, AVMediaType, AVPixelFormat,
};

use mcai_worker_sdk::job::JobResult;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
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
  response_sender: Option<Arc<Mutex<Sender<ProcessResult>>>>,
  frame_count: AtomicU32,
  sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WorkerParameters {
  /// Source path
  source_path: String,
  // /// The OCR result file path
  destination_path: String,
  /// The language to be detected
  language: Option<String>,
  /// The part of the frame to focus on
  region_of_interest: Option<RegionOfInterest>,
  /// The video sampling rate (default: 1)
  sample_rate: Option<u32>,
  /// Expected image width
  width: Option<u32>,
  /// Expected image height
  height: Option<u32>,
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

  fn init_process(
    &mut self,
    parameters: WorkerParameters,
    format_context: Arc<Mutex<FormatContext>>,
    response_sender: Arc<Mutex<Sender<ProcessResult>>>,
  ) -> Result<Vec<StreamDescriptor>, MessageError> {
    self.language = parameters.language.unwrap_or_else(|| "eng".to_string());
    self.response_sender = Some(response_sender);
    self.sample_rate = parameters.sample_rate;

    // get first video stream index
    let format_context = format_context.lock().unwrap();

    for stream_index in 0..format_context.get_nb_streams() {
      if format_context.get_stream_type(stream_index as isize) == AVMediaType::AVMEDIA_TYPE_VIDEO {
        let scaling = match (parameters.width, parameters.height) {
          (None, None) => None,
          (width, height) => Some(Scaling { width, height }),
        };

        let mut video_filters = vec![];
        if let Some(region_of_interest) = parameters.region_of_interest {
          video_filters.push(VideoFilter::Crop(region_of_interest));
        }

        if let Some(scaling) = scaling {
          video_filters.push(VideoFilter::Resize(scaling));
        }

        video_filters.push(VideoFilter::Format(VideoFormat {
          pixel_formats: "rgb24".to_string(),
        }));

        let stream_descriptor = StreamDescriptor::new_video(stream_index as usize, video_filters);

        return Ok(vec![stream_descriptor]);
      }
    }
    Err(MessageError::RuntimeError(
      "Missing video stream in the source".to_string(),
    ))
  }

  fn process_frame(
    &mut self,
    job_result: JobResult,
    _stream_index: usize,
    frame: Frame,
  ) -> Result<ProcessResult, MessageError> {
    let frame_count = self.frame_count.fetch_add(1, Ordering::Relaxed);
    if let Some(sample_rate) = self.sample_rate {
      if frame_count % sample_rate != 0 {
        return Ok(ProcessResult::empty());
      }
    }

    let recognised_text = unsafe {
      let pixel_format = std::mem::transmute::<_, AVPixelFormat>((*frame.frame).format);

      let av_pix_fmt_desc = av_pix_fmt_desc_get(pixel_format);
      let bytes_per_pixel = av_get_bits_per_pixel(av_pix_fmt_desc) / 8;

      let width = (*frame.frame).width;
      let height = (*frame.frame).height;
      let linesize = (*frame.frame).linesize[0];

      let buffer_size = (linesize * height) as usize;

      let data: Vec<u8> = Vec::from_raw_parts((*frame.frame).data[0], buffer_size, buffer_size);

      trace!(
        "Process OCR for frame {}: width={}, height={}, linesize={}",
        frame_count,
        width,
        height,
        linesize
      );
      let text = tesseract::ocr_from_frame(
        &data,
        width,
        height,
        bytes_per_pixel,
        linesize,
        &self.language,
      )
      .unwrap();
      trace!(target: &job_result.get_str_job_id(), "{:?}", text);

      std::mem::forget(data);

      RecognisedText {
        pts: (*frame.frame).pts as u64,
        text,
      }
    };

    Ok(ProcessResult::new_json(recognised_text))
  }

  fn ending_process(&mut self) -> Result<(), MessageError> {
    if let Some(sender) = &self.response_sender {
      sender
        .lock()
        .unwrap()
        .send(ProcessResult::end_of_process())
        .unwrap();
    }
    Ok(())
  }
}

fn main() {
  let worker = TextRecognitionEvent::default();
  start_worker(worker);
}
