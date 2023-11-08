document.addEventListener("DOMContentLoaded", function () {
  // Получаем ссылку на кнопку "View More" и на содержимое слайда
  const viewMoreButton = document.querySelector(".card:nth-child(1) .button");
  const slideContent = document.querySelector(
    ".card:nth-child(1) .card-content"
  );

  // Добавляем обработчик события на клик по кнопке
  viewMoreButton.addEventListener("click", function () {
    // Изменяем содержимое слайда (например, текст)
    slideContent.innerHTML = `
        <h2 class="name">David Dell</h2>
        <p class="description">
          This is the updated content of the first slide. You clicked "View More".
        </p>
        <button class="button">View More</button>
      `;

    // Можно также выполнить другие действия или анимации по желанию
  });
});
