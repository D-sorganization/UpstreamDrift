//! Error types for the URDF/MJCF crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UrdfError {
    #[error("invalid XML: {0}")]
    Xml(String),
    #[error("invalid URDF: {0}")]
    Schema(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("write error: {0}")]
    Write(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<quick_xml::Error> for UrdfError {
    fn from(value: quick_xml::Error) -> Self {
        UrdfError::Xml(value.to_string())
    }
}

impl From<quick_xml::events::attributes::AttrError> for UrdfError {
    fn from(value: quick_xml::events::attributes::AttrError) -> Self {
        UrdfError::Xml(value.to_string())
    }
}

impl From<std::str::Utf8Error> for UrdfError {
    fn from(value: std::str::Utf8Error) -> Self {
        UrdfError::Xml(value.to_string())
    }
}

pub type UrdfResult<T> = Result<T, UrdfError>;
