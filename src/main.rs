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

#[derive(Debug, Deserialize)]
pub struct SpookyAuthor {
    id: String,
    text: String,
    author: String
}

impl SpookyAuthor {
    pub fn into_tokens(&self) -> String {
        let lc_text = self.text.to_lowercase();
        let tok = VTextTokenizer::new("en");
        let tokens: Vec<&str> = tok.tokenize(lc_text.as_str()).collect();
        let en_stemmer = Stemmer::create(Algorithm::English);
        
        let tokens: Vec<String> = tokens.iter()
            .map(|x| en_stemmer.stem(x).into_owned())
            .collect();
        
        let mut tokens: Vec<&str> = tokens.iter()
            .map(|x| x.as_str())
            .collect();
        
        let stops: HashSet<_> = Spark::stopwords(Language::English)
            .unwrap()
            .iter()
            .collect();
        
        tokens.retain(|s| !stops.contains(s));

        tokens.iter().join(" ")
    }
}