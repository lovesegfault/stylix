extern crate image;
use image::error::ImageError;
use image::io::Reader as ImageReader;

extern crate itertools;
use itertools::Itertools;

extern crate palette;
use palette::{ColorDifference, IntoColor, Lab, Pixel, Srgb};
use palette::convert::FromColorUnclamped;
use palette::rgb::Rgb;

extern crate ordered_float;
use ordered_float::NotNan;

extern crate quick_error;
use quick_error::quick_error;

extern crate rand;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

extern crate rand_chacha;
use rand_chacha::ChaCha8Rng;

extern crate serde_json;

use std::cmp::min;
use std::collections::HashMap;
use std::env;
use std::io;
use std::process::exit;
use std::str::FromStr;

// How many palettes to generate for each generation.
const POPULATION_SIZE: usize = 1000;
// How many palettes to pass through the next generation.
const SURVIVORS: usize = 100;
// Chance of a color being randomised between generations.
const MUTATION_PROBABILITY: f32 = 0.1;
// When the fitness score improves by less than this value, terminate the algorithm.
const CHANGE_THRESHOLD: f32 = 0.01;

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        IOError(error: io::Error) {
            from(error: io::Error) -> (error)
            display("IO error: {}", error)
        }
        ImageError(error: ImageError) {
            from(error: ImageError) -> (error)
            display("error loading image: {}", error)
        }
        InvalidPolarity(given: String) {
            display("unexpected polarity: {}", given)
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Polarity {
    Light,
    Dark,
    Either
}

impl Polarity {
    fn lightness_error(&self, palette: &Palette<Lab>) -> NotNan<f32> {
        match self {
            Polarity::Light => NotNan::new(
                (palette.0[0].l - 90.0).abs() +
                (palette.0[1].l - 70.0).abs() +
                (palette.0[2].l - 55.0).abs() +
                (palette.0[3].l - 35.0).abs() +
                (palette.0[4].l - 25.0).abs() +
                (palette.0[5].l - 10.0).abs() +
                (palette.0[6].l - 5.0).abs() +
                (palette.0[7].l - 5.0).abs() +
                palette.accent().iter()
                    .map(|color| (color.l - 40.0).abs())
                    .sum::<f32>()
            ).unwrap(),

            Polarity::Dark => NotNan::new(
                (palette.0[0].l - 10.0).abs() +
                (palette.0[1].l - 30.0).abs() +
                (palette.0[2].l - 45.0).abs() +
                (palette.0[3].l - 65.0).abs() +
                (palette.0[4].l - 75.0).abs() +
                (palette.0[5].l - 90.0).abs() +
                (palette.0[6].l - 95.0).abs() +
                (palette.0[7].l - 95.0).abs() +
                palette.accent().iter()
                    .map(|color| (color.l - 60.0).abs())
                    .sum::<f32>()
            ).unwrap(),

            Polarity::Either => min(
                Polarity::Light.lightness_error(palette),
                Polarity::Dark.lightness_error(palette)
            )
        }
    }
}

impl FromStr for Polarity {
    type Err = Error;
    fn from_str(input: &str) -> Result<Polarity, Self::Err> {
        match input {
            "light" => Ok(Polarity::Light),
            "dark" => Ok(Polarity::Dark),
            "either" => Ok(Polarity::Either),
            _ => Err(Error::InvalidPolarity(input.to_string()))
        }
    }
}

#[derive(Debug)]
pub struct Palette<P>(Vec<P>);

impl<P> Palette<P> {
    fn primary(&self) -> &[P] {
        &self.0[0..8]
    }

    fn accent(&self) -> &[P] {
        &self.0[8..16]
    }
}

impl<P: Clone> Palette<P> {
    fn create_random<R: Rng>(rng: &mut R, pixels: &[P]) -> Self {
        Palette(
            pixels.choose_multiple(rng, 16)
                .map(|pixel| (*pixel).clone())
                .collect()
        )
    }
}

impl Palette<Lab> {
    fn fitness(&self, polarity: &Polarity) -> NotNan<f32> {
        let primary_similarity =
            self.primary().iter()
            .cartesian_product(self.primary().iter())
            .map(|(a, b)| NotNan::new(a.get_color_difference(b)).unwrap())
            .max().unwrap();

        let accent_difference =
            self.accent().iter()
            .cartesian_product(self.accent().iter())
            .map(|(a, b)| NotNan::new(a.get_color_difference(b)).unwrap())
            .min().unwrap();

        let lightness_error = polarity.lightness_error(self);

        accent_difference - (primary_similarity / 10.0) - lightness_error
    }
}

fn grow_population<R: Rng, P: Clone>(
    rng: &mut R,
    survivors: &[Palette<P>]
) -> Vec<Palette<P>> {
    let mut population = Vec::with_capacity(POPULATION_SIZE);

    while population.len() < POPULATION_SIZE {
        let parent_a = survivors.choose(rng).unwrap();
        let parent_b = survivors.choose(rng).unwrap();

        // Randomly select colours from either parent
        let child = Palette(
            parent_a.0.iter()
            .zip(parent_b.0.iter())
            .map(|(a, b)| if rng.gen() { (*a).clone() } else { (*b).clone() })
            .collect()
        );

        population.push(child);
    }

    population
}

fn mutate_population<R: Rng, P: Clone>(
    pixels: &[P],
    rng: &mut R,
    population: &mut [Palette<P>]
) {
    for palette in population.iter_mut() {
        for pixel in palette.0.iter_mut() {
            if rng.gen::<f32>() < MUTATION_PROBABILITY {
                *pixel = (*pixels.choose(rng).unwrap()).clone();
            }
        }
    }
}

fn evolve_population<R: Rng>(
    polarity: &Polarity,
    pixels: &[Lab],
    rng: &mut R,
    mut population: Vec<Palette<Lab>>,
    old_max_fitness: NotNan<f32>
) -> Vec<Palette<Lab>> {
    population.sort_by_cached_key(|palette| palette.fitness(polarity));

    let max_fitness = population.last().unwrap().fitness(polarity);

    let change = NotNan::new(1.0).unwrap() - (max_fitness / old_max_fitness);
    if change < NotNan::new(CHANGE_THRESHOLD).unwrap() {
        eprintln!("Stopping with fitness {}", max_fitness);

        population
    } else {
        eprintln!("Continuing with fitness {}", max_fitness);

        let survivors = population.split_off(population.len() - SURVIVORS);
        let mut new_population = grow_population(rng, &survivors);
        mutate_population(pixels, rng, &mut new_population);

        evolve_population(polarity, pixels, rng, new_population, max_fitness)
    }
}

pub fn generate(polarity: &Polarity, image_path: &str) -> Result<Palette<Lab>, Error> {
    let pixels: Vec<Lab> =
        // Read the image file
        ImageReader::open(image_path)?
        .decode()?
        // The LAB colour space requires a float data type
        .into_rgb32f()
        .pixels()
        .map(|pixel| {
            // Convert each pixel from `image::Rgb` to `palette::Lab`
            let pixel: Srgb = *Srgb::from_raw(&pixel.0);
            pixel.into_color()
        })
        .collect();

    // Random number generation must be reproducible so as to not change the colour scheme for
    // every NixOS rebuild, so we seed it with 0
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    let initial_population: Vec<Palette<Lab>> =
        (0..POPULATION_SIZE)
        .map(|_| Palette::create_random(&mut rng, &pixels))
        .collect();

    let mut population = evolve_population(
        polarity, &pixels, &mut rng, initial_population,
        NotNan::new(-f32::INFINITY).unwrap()
    );

    Ok(population.pop().unwrap())
}

pub fn print_palette<P: Clone>(palette: &Palette<P>) -> serde_json::Result<()>
where Rgb: FromColorUnclamped<P> {
    let mut output_table: HashMap<String, String> = HashMap::with_capacity(16);

    for (base, color) in palette.0.iter().enumerate() {
        let rgb: Rgb = (*color).clone().into_color();
        let rgb: Rgb<_, u8> = rgb.into_format();

        output_table.insert(
            format!("base{:02X}", base),
            format!("{:x}", rgb)
        );
    }

    serde_json::to_writer(io::stdout(), &output_table)
}

pub fn main() {
    if let (Some(polarity), Some(image_path)) = (env::args().nth(1), env::args().nth(2)) {
        let optimal_palette = generate(&Polarity::from_str(&polarity).unwrap(), &image_path).unwrap();
        print_palette(&optimal_palette).unwrap();
    } else {
        eprintln!("Required arguments: «polarity» «image path»");
        exit(1);
    }
}
