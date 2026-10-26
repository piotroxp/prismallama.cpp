import { loadAppRoute } from "$lib/stores/init";
import type { PageLoad } from "./$types";

export const load: PageLoad = ({ fetch }) => loadAppRoute(fetch);
