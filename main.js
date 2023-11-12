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

  nextSlide.addEventListener(
    "transitionend",
    function handleTransitionEnd() {
      currentSlide.style.transform = "";
      currentSlide.style.opacity = "";
      nextSlide.style.transform = "";
      nextSlide.style.opacity = "";

      nextSlide.removeEventListener("transitionend", handleTransitionEnd);

      currentSlideIndex = nextIndex;
      isAnimating = false;
    },
    { once: true }
  );
}

document.addEventListener("wheel", handleScroll);
