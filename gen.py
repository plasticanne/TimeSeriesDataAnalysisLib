import pdoc

Module.html()


pdoc.html(self.import_path_from_req_url,
                         reload=True, http_server=True, external_links=True,
                         **self.template_config)