use mcai_worker_sdk::job::{Job, JobResult, JobStatus};
use mcai_worker_sdk::{debug, trace};
use mcai_worker_sdk::{McaiChannel, MessageError, ParametersContainer};
use serde::Serialize;
use stainless_ffmpeg::{
  filter_graph::FilterGraph, format_context::FormatContext, order::filter::Filter,
  order::filter_output::FilterOutput, order::ParameterValue, packet::Packet,
  video_decoder::VideoDecoder,
};
use stainless_ffmpeg_sys::{
  av_get_bits_per_pixel, av_init_packet, av_packet_alloc, av_pix_fmt_desc_get, av_read_frame,
  AVPixelFormat,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Error;
use std::io::Write;
use std::mem;
use std::time::Instant;

pub const SOURCE_PATH_PARAMETER: &str = "source_path";
pub const LANGUAGE_PARAMETER: &str = "language";
pub const DESTINATION_PATH_PARAMETER: &str = "destination_path";
pub const SAMPLE_RATE_PARAMETER: &str = "sample_rate";

#[derive(Debug, Serialize)]
pub struct FrameAnalysis {
  #[serde(rename = "Coords")]
  coordinates: (u32, u32, u32, u32),
  #[serde(rename = "Confidence")]
  confidence: String,
  #[serde(rename = "Frame")]
  frame: usize,
  #[serde(rename = "Text")]
  text: String,
}

pub fn process(
  _channel: Option<McaiChannel>,
  job: &Job,
  job_result: JobResult,
) -> Result<JobResult, MessageError> {
  let source_path = get_required_string_parameter_value(job, &job_result, SOURCE_PATH_PARAMETER)?;
  let language = get_required_string_parameter_value(job, &job_result, LANGUAGE_PARAMETER)?;
  let destination_path =
    get_required_string_parameter_value(job, &job_result, DESTINATION_PATH_PARAMETER)?;

  let sample_rate = job
    .get_parameter::<i64>(SAMPLE_RATE_PARAMETER)
    .unwrap_or(1);

  let result = apply_ocr(&source_path, &language, sample_rate as usize, None).map_err(|error| {
    MessageError::ProcessingError(
      job_result
        .clone()
        .with_status(JobStatus::Error)
        .with_message(&error),
    )
  })?;

  to_file(&destination_path, &result)
    .map_err(|error| MessageError::from(error, job_result.clone()))?;

  Ok(job_result.with_status(JobStatus::Completed))
}

fn get_required_string_parameter_value(
  job: &Job,
  job_result: &JobResult,
  parameter_key: &str,
) -> Result<String, MessageError> {
  job.get_parameter::<String>(parameter_key).map_err(|e| {
    MessageError::ProcessingError(
      job_result
        .clone()
        .with_status(JobStatus::Error)
        .with_message(&format!(
          "Invalid job message: missing expected '{}' parameter: {:?}",
          parameter_key,
          e
        )),
    )
  })
}

fn apply_ocr(filename: &str, language: &str, sample_rate: usize, coordinates: Option<(u32, u32, u32, u32)>) -> Result<String, String> {
  let mut context = FormatContext::new(filename)?;
  context.open_input()?;

  let video_decoder = VideoDecoder::new("h264".to_string(), &context, 0)?;

  let mut graph = FilterGraph::new()?;

  graph.add_input_from_video_decoder("video_input", &video_decoder)?;
  graph.add_video_output("video_output")?;

  let format_filter_parameters: HashMap<String, ParameterValue> = [("pix_fmts", "rgb24")]
    .iter()
    .cloned()
    .map(|(key, value)| (key.to_string(), ParameterValue::String(value.to_string())))
    .collect();

  let format_filter_definition = Filter {
    name: "format".to_string(),
    label: Some("format_filter".to_string()),
    parameters: format_filter_parameters,
    inputs: None,
    outputs: Some(vec![FilterOutput {
      stream_label: "video_output".to_string(),
    }]),
  };

  let format_filter = graph.add_filter(&format_filter_definition)?;

  if let Some((h1, h2, w1, w2)) = coordinates {
    let out_w = h2 - h1;
    let out_h = w2 - w1;

    let crop_filter_parameters: HashMap<String, ParameterValue> = [
      ("out_w", out_w.to_string()),
      ("out_h", out_h.to_string()),
      ("x", w1.to_string()),
      ("y", h1.to_string()),
    ]
      .iter()
      .cloned()
      .map(|(key, value)| (key.to_string(), ParameterValue::String(value)))
      .collect();

    let crop_filter_definition = Filter {
      name: "crop".to_string(),
      label: Some("crop_filter".to_string()),
      parameters: crop_filter_parameters,
      inputs: None,
      outputs: None,
    };
    let crop_filter = graph.add_filter(&crop_filter_definition)?;

    graph.connect_input("video_input", 0, &crop_filter, 0)?;
    graph.connect(&crop_filter, 0, &format_filter, 0)?;
  } else {
    graph.connect_input("video_input", 0, &format_filter, 0)?;
  }

  graph.connect_output(&format_filter, 0, "video_output", 0)?;

  graph.validate()?;

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
            if frame_count % sample_rate != 0 {
              frame_count += 1;
              continue;
            }

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

            let frame_width = (*video_frame.frame).width;
            let frame_height = (*video_frame.frame).height;

            let result = tesseract::ocr_from_frame(
              &data,
              frame_width,
              frame_height,
              bytes_per_pixel,
              (*video_frame.frame).linesize[0],
              language,
            );

            debug!("Result computed in {} ms:", chrono.elapsed().as_millis());
            trace!("{}", result);

            let coords =
              if let Some(coords) = coordinates {
                coords
              } else {
                (0, frame_height as u32, 0, frame_width as u32)
              };

            text_analysis.push(FrameAnalysis {
              coordinates: coords,
              confidence: "NA".to_string(),
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

fn to_file(destination_path: &str, ocr_result: &str) -> Result<(), Error> {
  let mut output_file = File::create(destination_path)?;
  output_file.write_all(ocr_result.as_bytes())?;
  Ok(())
}
