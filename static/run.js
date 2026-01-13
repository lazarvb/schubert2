document.addEventListener('DOMContentLoaded', () => {
    let dotsInterval;

    function startDotsAnimation() {
        const dots = document.getElementById('dots');
        let count = 0;
        dotsInterval = setInterval(() => {
            count = (count + 1) % 4;
            dots.textContent = '.'.repeat(count);
        }, 500);
    }

    function stopDotsAnimation() {
        clearInterval(dotsInterval);
        const dots = document.getElementById('dots');
        if (dots) dots.textContent = '';
    }

    const menuBtn = document.querySelector('.menu-btn');
    const dropMenu = document.getElementById('menu-dropdown');

    const showMenu = () => dropMenu.classList.add('show');
    const hideMenu = () => {
        setTimeout(() => {
            if (!menuBtn.matches(':hover') && !dropMenu.matches(':hover')) {
                dropMenu.classList.remove('show');
            }
        }, 100);
    };

    [menuBtn, dropMenu].forEach((el) => {
        el.addEventListener('mouseenter', showMenu);
        el.addEventListener('mouseleave', hideMenu);
    });

    const menuLinks = dropMenu.querySelectorAll('a');

    menuLinks.forEach((link) => {
        const img = link.querySelector('img');
        const originalSrc = img.src;
        const hoverSrc = link.dataset.hover;

        link.addEventListener('mouseenter', () => {
            img.src = hoverSrc;
        });

        link.addEventListener('mouseleave', () => {
            img.src = originalSrc;
        });
    });

    const fileInput = document.getElementById('file');
    const chooseBtn = document.getElementById('choose-file-btn');
    const submitBtn = document.getElementById('submit-btn');
    const sliderContainer = document.getElementById('slider-container');

    chooseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        const show = fileInput.files.length > 0;

        submitBtn.style.display = show ? 'inline-block' : 'none';
        sliderContainer.style.display = show ? 'block' : 'none';
    });

    const submitText = document.getElementById('submit-text');

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];

        if (file) {
            submitText.innerHTML = `Upload & Process : <br><br> ${file.name}`;
        } else {
            submitText.textContent = 'Upload & Process';
        }
    });

    const form = document.getElementById('upload-form');
    const errorDiv = document.getElementById('error-message');

    if (form && fileInput && errorDiv) {
        form.addEventListener('submit', async function (e) {
            e.preventDefault();
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) {
                loadingMessage.style.display = 'block';
                startDotsAnimation();
            }
            submitBtn.disabled = true;

            errorDiv.innerText = '';

            const file = fileInput.files[0];

            if (!file) {
                errorDiv.innerText = 'Aucun fichier sélectionné.';
                return;
            }

            const slider = document.getElementById('nbarline');

            const formData = new FormData();
            formData.append('file', file);
            formData.append('nbarline', slider.value);

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                stopDotsAnimation();
                loadingMessage.style.display = 'none';
                submitBtn.disabled = false;

                if (!data.success) {
                    errorDiv.innerText = data.message;
                } else {
                    window.location.href = data.download_url;
                }
            } catch (error) {
                stopDotsAnimation();
                loadingMessage.style.display = 'none';
                submitBtn.disabled = false;
                errorDiv.innerText = 'Erreur serveur.';
            }
        });
    }

    gsap.registerPlugin(SplitText, ScrambleTextPlugin);

    const root = document.querySelector('#scrambled');
    if (!root) return;

    const split = new SplitText(root.querySelector('p'), {
        type: 'chars',
        charsClass: 'char',
    });
    const chars = split.chars;

    const radius = 100;
    const duration = 1.2;
    const speed = 0.5;
    const scrambleChars = '.:';

    chars.forEach((c) => {
        gsap.set(c, {
            display: 'inline-block',
            attr: { 'data-content': c.innerHTML },
        });
    });

    const handleMove = (e) => {
        chars.forEach((c) => {
            const rect = c.getBoundingClientRect();
            const dx = e.clientX - (rect.left + rect.width / 2);
            const dy = e.clientY - (rect.top + rect.height / 2);
            const dist = Math.hypot(dx, dy);

            if (dist < radius) {
                gsap.to(c, {
                    overwrite: true,
                    duration: duration * (1 - dist / radius),
                    scrambleText: {
                        text: c.dataset.content || '',
                        chars: scrambleChars,
                        speed,
                    },
                    ease: 'none',
                });
            }
        });
    };

    root.addEventListener('pointermove', handleMove);
});
