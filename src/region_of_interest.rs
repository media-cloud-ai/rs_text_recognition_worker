use mcai_worker_sdk::ParameterValue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegionOfInterest {
  top: Option<u32>,
  left: Option<u32>,
  right: Option<u32>,
  bottom: Option<u32>,
  width: Option<u32>,
  height: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Coordinates {
  pub top: u32,
  pub left: u32,
  pub width: u32,
  pub height: u32,
}

impl ParameterValue for RegionOfInterest {
  fn get_type_as_string() -> String {
    "region_of_interest".to_string()
  }
}

impl RegionOfInterest {
  pub fn get_coordinates(&self) -> Result<Coordinates, String> {
    match self.clone() {
      RegionOfInterest {
        top: Some(top),
        left: Some(left),
        right: Some(right),
        bottom: Some(bottom),
        width: None,
        height: None,
      } => Ok(Coordinates {
        top,
        left,
        width: right - left,
        height: bottom - top,
      }),
      RegionOfInterest {
        top: Some(top),
        left: Some(left),
        right: None,
        bottom: None,
        width: Some(width),
        height: Some(height),
      } => Ok(Coordinates {
        top,
        left,
        width,
        height,
      }),
      RegionOfInterest {
        top: Some(top),
        left: Some(left),
        right: None,
        bottom: Some(bottom),
        width: Some(width),
        height: None,
      } => Ok(Coordinates {
        top,
        left,
        width,
        height: bottom - top,
      }),
      RegionOfInterest {
        top: Some(top),
        left: Some(left),
        right: Some(right),
        bottom: None,
        width: None,
        height: Some(height),
      } => Ok(Coordinates {
        top,
        left,
        width: right - left,
        height,
      }),
      RegionOfInterest {
        top: None,
        left: Some(left),
        right: None,
        bottom: Some(bottom),
        width: Some(width),
        height: Some(height),
      } => Ok(Coordinates {
        top: bottom - height,
        left,
        width,
        height,
      }),
      RegionOfInterest {
        top: Some(top),
        left: None,
        right: Some(right),
        bottom: None,
        width: Some(width),
        height: Some(height),
      } => Ok(Coordinates {
        top,
        left: right - width,
        width,
        height,
      }),
      RegionOfInterest {
        top: None,
        left: None,
        right: Some(right),
        bottom: Some(bottom),
        width: Some(width),
        height: Some(height),
      } => Ok(Coordinates {
        top: bottom - height,
        left: right - width,
        width,
        height,
      }),
      _ => Err(format!(
        "Cannot compute coordinates from such a region of interest: {:?}",
        self
      )),
    }
  }
}

#[test]
pub fn region_of_interest_to_coordinates_top_left_right_bottom() {
  let region_of_interest = RegionOfInterest {
    top: Some(0),
    left: Some(0),
    right: Some(200),
    bottom: Some(100),
    width: None,
    height: None,
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_top_left_width_height() {
  let region_of_interest = RegionOfInterest {
    top: Some(0),
    left: Some(0),
    right: None,
    bottom: None,
    width: Some(200),
    height: Some(100),
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_top_left_bottom_width() {
  let region_of_interest = RegionOfInterest {
    top: Some(0),
    left: Some(0),
    right: None,
    bottom: Some(100),
    width: Some(200),
    height: None,
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_top_left_right_height() {
  let region_of_interest = RegionOfInterest {
    top: Some(0),
    left: Some(0),
    right: Some(200),
    bottom: None,
    width: None,
    height: Some(100),
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_left_bottom_width_height() {
  let region_of_interest = RegionOfInterest {
    top: None,
    left: Some(0),
    right: None,
    bottom: Some(100),
    width: Some(200),
    height: Some(100),
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_top_right_width_height() {
  let region_of_interest = RegionOfInterest {
    top: Some(0),
    left: None,
    right: Some(200),
    bottom: None,
    width: Some(200),
    height: Some(100),
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}

#[test]
pub fn region_of_interest_to_coordinates_right_bottom_width_height() {
  let region_of_interest = RegionOfInterest {
    top: None,
    left: None,
    right: Some(200),
    bottom: Some(100),
    width: Some(200),
    height: Some(100),
  };

  let coordinates = region_of_interest.get_coordinates().unwrap();

  assert_eq!(0, coordinates.top);
  assert_eq!(0, coordinates.left);
  assert_eq!(200, coordinates.width);
  assert_eq!(100, coordinates.height);
}
