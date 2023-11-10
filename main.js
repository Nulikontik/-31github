const slider = document.querySelector(".swiper");
const slides = document.querySelectorAll(".swiper-slide");
let currentSlideIndex = 0;
let isAnimating = false;

function handleScroll(event) {
  if (isAnimating) return;

  const delta = event.deltaY;
  if (delta > 0 && currentSlideIndex < slides.length - 1) {
    animateSlide(currentSlideIndex + 1);
  } else if (delta < 0 && currentSlideIndex > 0) {
    animateSlide(currentSlideIndex - 1);
  }
}

function animateSlide(nextIndex) {
  isAnimating = true;
  const currentSlide = slides[currentSlideIndex];
  const nextSlide = slides[nextIndex];

  nextSlide.style.transform = "scale(1.1)";
  nextSlide.style.opacity = "1";

  setTimeout(function () {
    currentSlide.style.transform = "scale(0.9)";
    currentSlide.style.opacity = "0";
    nextSlide.style.transform = "";
    nextSlide.style.opacity = "";
    currentSlideIndex = nextIndex;
    isAnimating = false;
  }, 600);
}

document.addEventListener("wheel", handleScroll);
