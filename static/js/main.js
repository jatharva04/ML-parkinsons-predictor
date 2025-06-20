window.onload = function () {
  const resultElement = document.getElementById('result');
  if (resultElement && resultElement.innerHTML.trim() !== '') {
    resultElement.scrollIntoView({ behavior: 'smooth' });
  }
};

// Spinner logic
document.addEventListener('DOMContentLoaded', function () {
  const form = document.querySelector('form');
  const spinner = document.getElementById('spinner');

  if (form && spinner) {
    form.addEventListener('submit', function () {
      spinner.style.display = 'block';
    });
  }
});
