extern crate serde;

#[macro_use]
extern crate serde_derive;

use std::io;
use std::vec::Vec;
use std::error::Error;
use std::io::Write;
use std::fs::File;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use fasttext::{ FastText, Args, ModelName, LossName };
use stopwords;
use std::collections::HashSet;
use stopwords::{ Spark, Language, Stopwords };
use itertools::Itertools;
use vtext::tokenize::VTextTokenizer;
use rust_stemmer::{ Algorithm, Stemmer };

