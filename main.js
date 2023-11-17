// Импортируем gsap
import { gsap } from "gsap";

// Используем gsap для создания анимации
var mySwiper = new Swiper(".swiper-container", {
  direction: "vertical",
  loop: true,
  pagination: ".swiper-pagination",
  grabCursor: true,
  speed: 2000,
  paginationClickable: true,
  parallax: true,
  autoplay: false,
  effect: "slide",
  mousewheelControl: 2,
  on: {
    slideChange: function () {
      // Анимация элементов при изменении слайда
      gsap.fromTo(
        ".swiper-slide-active",
        { opacity: 0 },
        { opacity: 1, duration: 0.5 }
      );

      // Добавьте дополнительные анимации для других элементов при необходимости
    },
  },
});
