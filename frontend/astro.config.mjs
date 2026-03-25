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
						{ label: 'Framework', slug: 'docs/guide/framework' },
						{ label: 'Adding a Task', slug: 'docs/guide/tasks' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{ label: 'SoftmaxGLMHMM', slug: 'docs/api/model' },
						{ label: 'Tasks API', slug: 'docs/api/tasks' },
						{ label: 'Postprocessing', slug: 'docs/api/postprocess' },
						{ label: 'Views', slug: 'docs/api/views' },
						{ label: 'Common Plots', slug: 'docs/api/common-plots' },
						{ label: 'Task Plots', slug: 'docs/api/task-plots' },
					],
				},
				{
					label: 'Analysis',
					items: [
						{ label: 'Interactive Notebooks', slug: 'docs/notebooks' },
					],
				},
			],
		}),
	],
});
