// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import icon from 'astro-icon';

// https://astro.build/config
export default defineConfig({
	site: 'https://tfm.javirm.com',
	integrations: [
		icon(),
		starlight({
			title: 'glmhmmt',
			description: 'A Softmax GLM-HMM built on Dynamax — documentation & API reference',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/javirm3/TFM' },
			],
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'docs/intro' },
						{ label: 'Quickstart', slug: 'docs/guide/quickstart' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{ label: 'SoftmaxGLMHMM', slug: 'docs/api/model' },
						{ label: 'Features', slug: 'docs/api/features' },
						{ label: 'Postprocessing', slug: 'docs/api/postprocess' },
						{ label: 'Views', slug: 'docs/api/views' },
					],
				},
			],
		}),
	],
});
