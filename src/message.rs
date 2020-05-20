use mcai_worker_sdk::job::{Job, JobResult, JobStatus};
use mcai_worker_sdk::{debug, trace};
use mcai_worker_sdk::{McaiChannel, MessageError, ParametersContainer};

use stainless_ffmpeg::filter_graph::FilterGraph;
use stainless_ffmpeg::format_context::FormatContext;
use stainless_ffmpeg::order;
use stainless_ffmpeg::order::filter_output::FilterOutput;
use stainless_ffmpeg::order::ParameterValue;
use stainless_ffmpeg::packet::Packet;
use stainless_ffmpeg::video_decoder::VideoDecoder;
use stainless_ffmpeg_sys::{
  av_get_bits_per_pixel, av_init_packet, av_packet_alloc, av_pix_fmt_desc_get, av_read_frame,
  AVPixelFormat,
};
use std::collections::HashMap;
use std::mem;
use std::time::Instant;

use serde::Serialize;

pub const SOURCE_PATH_PARAMETER: &str = "source_path";
pub const LANGUAGE_PARAMETER: &str = "language";

#[derive(Debug, Serialize)]
pub struct FrameAnalysis {
  frame: usize,
  text: String,
}

pub fn process(
  _channel: Option<McaiChannel>,
  job: &Job,
  job_result: JobResult,
) -> Result<JobResult, MessageError> {
  let source_path = job
    .get_string_parameter(SOURCE_PATH_PARAMETER)
    .ok_or_else(|| {
      MessageError::ProcessingError(
        job_result
          .clone()
          .with_status(JobStatus::Error)
          .with_message(&format!(
            "Invalid job message: missing expected '{}' parameter.",
            SOURCE_PATH_PARAMETER
          )),
      )
    })?;

  let language = job
    .get_string_parameter(LANGUAGE_PARAMETER)
    .ok_or_else(|| {
      MessageError::ProcessingError(
        job_result
          .clone()
          .with_status(JobStatus::Error)
          .with_message(&format!(
            "Invalid job message: missing expected '{}' parameter.",
            LANGUAGE_PARAMETER
          )),
      )
    })?;

  let result = apply_ocr(&source_path, &language).map_err(|error| {
    MessageError::ProcessingError(
      job_result
        .clone()
        .with_status(JobStatus::Error)
        .with_message(&error),
    )
  })?;

  Ok(
    job_result
      .with_status(JobStatus::Completed)
      .with_message(&result),
  )
}

pub fn apply_ocr(filename: &str, language: &str) -> Result<String, String> {
  let mut context = FormatContext::new(filename).unwrap();
  context.open_input().unwrap();

  let video_decoder = VideoDecoder::new("h264".to_string(), &context, 0).unwrap();

  let mut graph = FilterGraph::new().unwrap();

  graph
    .add_input_from_video_decoder("video_input", &video_decoder)
    .unwrap();
  graph.add_video_output("video_output").unwrap();

  let parameters: HashMap<String, ParameterValue> = [("pix_fmts", "rgb24")]
    .iter()
    .cloned()
    .map(|(key, value)| (key.to_string(), ParameterValue::String(value.to_string())))
    .collect();

  let filter_definition = order::filter::Filter {
    name: "format".to_string(),
    label: Some("format_filter".to_string()),
    parameters,
    inputs: None,
    outputs: Some(vec![FilterOutput {
      stream_label: "video_output".to_string(),
    }]),
  };

  let filter = graph.add_filter(&filter_definition).unwrap();
  graph.connect_input("video_input", 0, &filter, 0).unwrap();
  graph.connect_output(&filter, 0, "video_output", 0).unwrap();

  graph.validate().unwrap();

  let mut text_analysis = Vec::new();
  let mut frame_count = 0 as usize;
  loop {
    unsafe {
      let av_packet = av_packet_alloc();
      av_init_packet(av_packet);
      if av_read_frame(context.format_context, av_packet) < 0 {
        debug!("No more packet to read.");
        break;
      } else {
        if (*av_packet).stream_index != 0 {
          continue;
        }

        let packet = Packet {
          name: None,
          packet: av_packet,
        };
        let frame = video_decoder.decode(&packet)?;

        if let Ok((_audio_frames, video_frames)) = graph.process(&[], &[frame]) {
          for video_frame in &video_frames {
            let buffer_size = (*video_frame.frame).linesize[0] * (*video_frame.frame).height;

            let av_pix_fmt_desc = av_pix_fmt_desc_get(AVPixelFormat::AV_PIX_FMT_RGB24);
            let bytes_per_pixel = av_get_bits_per_pixel(av_pix_fmt_desc) / 8;

            debug!(
              "{}: width={} height={} key_frame={} linesize={} format={}, bytes_per_pixel={} ==> buffer_size={}",
              frame_count,
              (*video_frame.frame).width,
              (*video_frame.frame).height,
              (*video_frame.frame).key_frame,
              (*video_frame.frame).linesize[0],
              (*video_frame.frame).format,
              bytes_per_pixel,
              buffer_size
            );

            let chrono = Instant::now();

            let data: Vec<u8> = Vec::from_raw_parts(
              (*video_frame.frame).data[0],
              buffer_size as usize,
              buffer_size as usize,
            );

            debug!("Start OCR with: data={}, language={}", data.len(), language);
            let result = tesseract::ocr_from_frame(
              &data,
              (*video_frame.frame).width,
              (*video_frame.frame).height,
              bytes_per_pixel,
              (*video_frame.frame).linesize[0],
              language,
            );

            debug!("Result computed in {} ms:", chrono.elapsed().as_millis());
            trace!("{}", result);

            text_analysis.push(FrameAnalysis {
              frame: frame_count,
              text: result,
            });

            mem::forget(data);
            frame_count += 1;
          }
        }
      }
    }
  }

  let json_result = serde_json::to_string(&text_analysis)
    .map_err(|error| format!("Unable to serialize OCR result: {:?}", error))?;
  Ok(json_result)
}
