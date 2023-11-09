// Получаем элементы слайдера и слайдов
var slider = document.querySelector(".swiper");
var slides = document.querySelectorAll(".swiper-slide");

// Устанавливаем начальные значения
var currentSlideIndex = 0;
var isAnimating = false;

// Функция для обработки событий скролла
function handleScroll(event) {
  // Проверяем, идет ли анимация
  if (isAnimating) return;

  // Определяем направление скролла
  var delta = event.deltaY;
  if (delta > 0 && currentSlideIndex < slides.length - 1) {
    // Прокрутка вниз
    animateSlide(currentSlideIndex + 1);
  } else if (delta < 0 && currentSlideIndex > 0) {
    // Прокрутка вверх
    animateSlide(currentSlideIndex - 1);
  }
}

// Функция для анимации перехода между слайдами
function animateSlide(nextIndex) {
  isAnimating = true;

  // Получаем текущий и следующий слайд
  var currentSlide = slides[currentSlideIndex];
  var nextSlide = slides[nextIndex];

  // Увеличиваем размер следующего слайда
  nextSlide.style.transform = "scale(1.1)";
  nextSlide.style.opacity = "1";

  // Запускаем анимацию
  setTimeout(function () {
    // Уменьшаем размер текущего слайда
    currentSlide.style.transform = "scale(0.9)";
    currentSlide.style.opacity = "0";

    // Сбрасываем стили для следующего слайда
    nextSlide.style.transform = "";
    nextSlide.style.opacity = "";

    // Обновляем текущий индекс слайда
    currentSlideIndex = nextIndex;
    isAnimating = false;
  }, 600); // Время анимации в миллисекундах
}

// Добавляем обработчик событий скролла
document.addEventListener("wheel", handleScroll);
