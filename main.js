const SLIDER_CLASS = ".swiper";
const SLIDE_CLASS = ".swiper-slide";

const slider = document.querySelector(SLIDER_CLASS);
const slides = document.querySelectorAll(SLIDE_CLASS);
let currentSlideIndex = 0;
let isAnimating = false;

function isWithinBounds(index) {
  return index >= 0 && index < slides.length;
}

function resetSlideStyles(slide) {
  slide.style.transform = "";
  slide.style.opacity = "";
}

function animateSlide(nextIndex) {
  isAnimating = true;
  const currentSlide = slides[currentSlideIndex];
  const nextSlide = slides[nextIndex];

  nextSlide.classList.add("active");
  nextSlide.classList.remove("inactive");

  nextSlide.addEventListener(
    "transitionend",
    () => {
      resetSlideStyles(currentSlide);

      currentSlide.classList.remove("active");
      currentSlide.classList.add("inactive");

      nextSlide.classList.remove("inactive");

      nextSlide.removeEventListener("transitionend", handleTransitionEnd);

      currentSlideIndex = nextIndex;
      isAnimating = false;
      window.removeEventListener("wheel", handleScroll);
      setTimeout(() => {
        window.addEventListener("wheel", handleScroll);
      }, 500); // Задержка включения скролла
    },
    { once: true }
  );
}

function handleTransitionEnd() {}

function handleScroll(event) {
  if (isAnimating) return;

  const delta = event.deltaY;
  const nextIndex = delta > 0 ? currentSlideIndex + 1 : currentSlideIndex - 1;

  if (isWithinBounds(nextIndex)) {
    animateSlide(nextIndex);
  }
}

window.addEventListener("wheel", handleScroll);
