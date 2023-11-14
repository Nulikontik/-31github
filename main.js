const slider = document.querySelector(".swiper");
const slides = document.querySelectorAll(".swiper-slide");
let currentSlideIndex = 0;
let isAnimating = false;
let enableScroll = true;

function isWithinBounds(index) {
  return index >= 0 && index < slides.length;
}

function resetSlideStyles(slide) {
  slide.style.transform = "";
  slide.style.opacity = "";
}

function animateSlide(nextIndex) {
  enableScroll = false;
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

      // В конце анимации установите флаг на true
      enableScroll = true;
    },
    { once: true }
  );
}

function handleTransitionEnd() {
  // Если в будущем появятся дополнительные действия по завершению перехода, их можно добавить сюда
}

function handleScroll(event) {
  if (isAnimating || !enableScroll) return;

  const delta = event.deltaY;
  const nextIndex = delta > 0 ? currentSlideIndex + 1 : currentSlideIndex - 1;

  if (isWithinBounds(nextIndex)) {
    animateSlide(nextIndex);
  }
}

window.addEventListener("wheel", handleScroll);
